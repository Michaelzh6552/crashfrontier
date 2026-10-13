from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from crashfrontier.config import ConicSettings, GridSettings, LPSettings
from crashfrontier.data.cross_section import CrossSection
from crashfrontier.errors import InfeasibleCrossSectionError
from crashfrontier.frontier import (
    Frontier,
    _depth_range,
    _validate_depth_enclosure,
    binding_quotes,
    compute_frontier,
    nested_frontiers,
    portfolio_bound,
    scenario_check,
)
from crashfrontier.geometry.polygon import point_in_convex_polygon
from tests.synthetic import scenarios, synthetic_quotes, true_targets

THRESHOLD = 0.90
SPREAD = 1e-6
SETTINGS = LPSettings()
FAST_GRID = GridSettings(levels=(16,), max_normalized_gap=0.08)
NESTED_GRID = GridSettings(levels=(12,), max_normalized_gap=0.08)
PHYSICAL_SETS = ("put_pair", "local_eight", "put_wing", "full_otm")
SUMMARY_KEYS = [
    "date",
    "expiry",
    "days",
    "am_settlement",
    "K",
    "gamma",
    "support_cap",
    "quote_set",
    "n_quotes",
    "n_strikes",
    "forward",
    "forward_source",
    "rate",
    "discount",
    "spot",
    "mode",
    "method",
    "status",
    "dimension",
    "prob_lo",
    "prob_hi",
    "loss_lo",
    "loss_hi",
    "depth_lo",
    "depth_hi",
    "area",
    "area_outer",
    "baseline_area",
    "area_ratio",
    "interior_ok",
    "interior_margin",
    "support_calls",
    "max_gap",
    "solver_gap",
    "elapsed_seconds",
]


def test_depth_range_does_not_apply_plotting_probability_floor() -> None:
    polygon = np.array(
        [[2e-5, 1.6e-5], [3e-5, 2.4e-5], [0.2, 0.02]],
        dtype=float,
    )
    low, high = _depth_range(polygon)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.8)


def test_large_depth_disagreement_is_marked_uncertified() -> None:
    assert not _validate_depth_enclosure(
        0.0,
        0.01,
        0.0,
        0.02,
        gamma=2,
        quote_set="custom",
    )


def _quotes(scenario_name: str = "bimodal", spread: float = SPREAD) -> pd.DataFrame:
    nodes, weights = scenarios(THRESHOLD)[scenario_name]
    return synthetic_quotes(nodes, weights, spread=spread)


def _section(quotes: pd.DataFrame) -> CrossSection:
    return CrossSection(
        date=pd.Timestamp("2021-07-21"),
        expiry=pd.Timestamp("2021-08-04"),
        am_settlement=0,
        root="SYN",
        days=14,
        tau=14.0 / 365.0,
        forward=1.0,
        forward_source="user",
        rate=0.0,
        rate_source="zero",
        discount=1.0,
        spot=float("nan"),
        quotes=quotes,
        filter_report={"kept": int(len(quotes))},
    )


def _cross_quote_inconsistent(quotes: pd.DataFrame) -> pd.DataFrame:
    index = int(np.argmin(quotes["norm_u"].to_numpy(dtype=float)))
    extra = quotes.iloc[[index]].copy()
    ask = float(extra["norm_u"].iloc[0])
    width = max(float(extra["norm_u"].iloc[0] - extra["norm_l"].iloc[0]), 1e-4)
    extra["norm_l"] = ask + 0.02
    extra["norm_u"] = ask + 0.02 + width
    extra["quote_id"] = str(extra["quote_id"].iloc[0]) + "_conflict"
    extra["bid"] = extra["norm_l"]
    extra["ask"] = extra["norm_u"]
    extra["norm_bid"] = extra["norm_l"]
    extra["norm_ask"] = extra["norm_u"]
    return pd.concat([quotes, extra], ignore_index=True)


def _wing_conflict(quotes: pd.DataFrame) -> pd.DataFrame:
    log_m = quotes["log_moneyness"].to_numpy(dtype=float)
    in_wing = (log_m >= -0.35) & (log_m <= 0.05)
    far = np.flatnonzero(in_wing & (np.abs(quotes["x"].to_numpy(dtype=float) - THRESHOLD) > 0.08))
    if far.size == 0:
        far = np.flatnonzero(in_wing)
    index = int(far[0])
    extra = quotes.iloc[[index]].copy()
    ask = float(extra["norm_u"].iloc[0])
    width = max(float(extra["norm_u"].iloc[0] - extra["norm_l"].iloc[0]), 1e-4)
    extra["norm_l"] = ask + 0.03
    extra["norm_u"] = ask + 0.03 + width
    extra["quote_id"] = str(extra["quote_id"].iloc[0]) + "_wing_conflict"
    extra["bid"] = extra["norm_l"]
    extra["ask"] = extra["norm_u"]
    extra["norm_bid"] = extra["norm_l"]
    extra["norm_ask"] = extra["norm_u"]
    return pd.concat([quotes, extra], ignore_index=True)


def _otm_call_conflict(quotes: pd.DataFrame) -> pd.DataFrame:
    extra = quotes.iloc[[0]].copy()
    extra["cp_flag"] = "C"
    extra["x"] = 1.10
    extra["strike"] = 1.10
    extra["log_moneyness"] = float(np.log(1.10))
    extra["role"] = "otm_call"
    extra["norm_l"] = 0.45
    extra["norm_u"] = 0.50
    extra["norm_bid"] = extra["norm_l"]
    extra["norm_ask"] = extra["norm_u"]
    extra["bid"] = extra["norm_l"]
    extra["ask"] = extra["norm_u"]
    extra["quote_id"] = "conflict_call"
    return pd.concat([quotes, extra], ignore_index=True)


def _centroid(points: np.ndarray) -> np.ndarray:
    cloud = np.atleast_2d(np.asarray(points, dtype=float))
    return cloud.mean(axis=0)


def test_compute_frontier_gamma0_is_two_dimensional() -> None:
    quotes = _quotes()
    frontier = compute_frontier(_section(quotes), THRESHOLD, quote_set=quotes)
    assert isinstance(frontier, Frontier)
    assert frontier.gamma == 0
    assert frontier.mode == "hard"
    assert frontier.method == "hull"
    assert frontier.dimension == 2
    assert frontier.status == "edges_certified"
    assert frontier.quote_set == "custom"
    assert len(frontier.inner) >= 3
    assert len(frontier.outer) >= 3
    for vertex in frontier.inner:
        assert point_in_convex_polygon(vertex, frontier.outer, tolerance=1e-7)
    assert frontier.area_outer + 1e-12 >= frontier.area
    assert frontier.diagnostics["elapsed_seconds"] > 0.0
    assert frontier.diagnostics["support_calls"] >= 8
    assert frontier.diagnostics["max_duality_gap"] < 1e-7
    assert frontier.summary()["solver_gap"] == pytest.approx(
        frontier.diagnostics["max_duality_gap"]
    )
    nodes, weights = scenarios(THRESHOLD)["bimodal"]
    probability, loss, _depth = true_targets(nodes, weights, THRESHOLD)
    assert frontier.contains(probability, loss) in {"inside", "boundary"}
    assert frontier.lo[0] - 2e-7 <= probability <= frontier.hi[0] + 2e-7
    assert frontier.lo[1] - 2e-7 <= loss <= frontier.hi[1] + 2e-7


def test_summary_keys_to_dict_and_contains() -> None:
    quotes = _quotes()
    frontier = compute_frontier(_section(quotes), THRESHOLD, quote_set=quotes)
    summary = frontier.summary()
    assert list(summary) == SUMMARY_KEYS
    payload = frontier.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["date"] == "2021-07-21"
    assert payload["expiry"] == "2021-08-04"
    assert isinstance(payload["inner"], list)
    assert isinstance(payload["normalizer"]["origin"], list)
    centroid = _centroid(frontier.inner)
    assert frontier.contains(float(centroid[0]), float(centroid[1])) == "inside"
    assert frontier.contains(0.99, 0.0) == "outside"
    text = frontier.describe()
    assert "quote_set=custom" in text
    assert "mode=hard" in text
    vertices = frontier.vertices()
    assert list(vertices.columns) == ["kind", "order", "prob", "loss", "depth"]
    assert set(vertices["kind"]) == {"inner", "outer"}
    planes = frontier.halfplane_table()
    assert list(planes.columns) == ["a_prob", "a_loss", "h"]
    assert len(planes) == len(frontier.halfplanes)
    values = frontier.inner @ frontier.halfplanes[:, :2].T
    assert np.max(values - frontier.halfplanes[:, 2]) <= 1e-6
    frontier.inner = np.array([[0.20, 0.01], [0.40, 0.01], [0.30, 0.05]], dtype=float)
    frontier.outer = np.array(
        [[0.10, 0.00], [0.50, 0.00], [0.50, 0.08], [0.10, 0.08]],
        dtype=float,
    )
    assert frontier.contains(0.30, 0.02) == "inside"
    assert frontier.contains(0.15, 0.02) == "boundary"
    assert frontier.contains(0.00, 0.00) == "outside"


def test_nested_frontiers_area_is_monotone(dataset) -> None:
    section = dataset.cross_section("2021-07-21", days=14)
    fronts = nested_frontiers(section, THRESHOLD)
    assert list(fronts) == ["put_pair", "local_eight", "put_wing"]
    areas = [fronts[name].area for name in fronts]
    assert all(fronts[name].mode == "hard" for name in fronts)
    assert all(fronts[name].dimension == 2 for name in fronts)
    assert all(fronts[name].gamma == 0 for name in fronts)
    assert areas[0] + 1e-10 >= areas[1] >= areas[2] - 1e-12
    for vertex in fronts["put_wing"].inner:
        assert point_in_convex_polygon(vertex, fronts["local_eight"].outer, tolerance=2e-7)
    strike_counts = [fronts[name].n_strikes for name in fronts]
    assert strike_counts[0] <= strike_counts[1] <= strike_counts[2]


def test_auto_mode_falls_back_to_relaxed(caplog: pytest.LogCaptureFixture) -> None:
    quotes = _cross_quote_inconsistent(_quotes())
    section = _section(quotes)
    with pytest.raises(InfeasibleCrossSectionError):
        compute_frontier(section, THRESHOLD, quote_set=quotes, mode="hard")
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    frontier = compute_frontier(section, THRESHOLD, quote_set=quotes, mode="auto")
    assert frontier.mode == "relaxed"
    assert frontier.diagnostics["relaxed"] is True
    assert frontier.diagnostics["budget"] > 0.0
    assert frontier.diagnostics["minimum_slack"] > 0.0
    assert not frontier.interior_ok
    assert np.isnan(frontier.interior_margin)
    assert "minimum slack budget" in caplog.text
    assert frontier.dimension in {1, 2}


def test_nested_frontiers_share_relaxed_budget() -> None:
    quotes = _wing_conflict(_quotes())
    fronts = nested_frontiers(_section(quotes), THRESHOLD)
    assert all(frontier.mode == "relaxed" for frontier in fronts.values())
    budgets = [float(frontier.diagnostics["budget"]) for frontier in fronts.values()]
    assert budgets[0] == pytest.approx(budgets[1])
    assert budgets[1] == pytest.approx(budgets[2])
    assert budgets[0] > 0.0
    areas = [fronts[name].area for name in ("put_pair", "local_eight", "put_wing")]
    assert areas[0] + 1e-8 >= areas[1] >= areas[2] - 1e-8


def test_degenerate_box_uses_axis_supports() -> None:
    quotes = _quotes()
    frontier = compute_frontier(
        _section(quotes),
        THRESHOLD,
        quote_set=quotes,
        lp_settings=LPSettings(degenerate_tolerance=1.0),
    )
    assert frontier.method == "degenerate"
    assert frontier.dimension in {0, 1}
    assert frontier.status in {"point", "segment"}
    assert frontier.area == 0.0
    assert frontier.area_outer == 0.0
    assert np.isnan(frontier.area_ratio)
    assert np.allclose(frontier.inner, frontier.outer)
    assert frontier.halfplanes.shape[1] == 3
    assert frontier.diagnostics["support_calls"] == 4


def test_support_cap_and_grid_method() -> None:
    quotes = _quotes()
    section = _section(quotes)
    capped = compute_frontier(section, THRESHOLD, quote_set=quotes, support_cap=5.0)
    assert capped.support_cap == 5.0
    assert capped.dimension == 2
    assert capped.mode == "hard"
    grid = compute_frontier(
        section,
        THRESHOLD,
        quote_set=quotes,
        method="grid",
        grid_settings=GridSettings(levels=(16,), max_normalized_gap=0.05),
    )
    assert grid.method == "grid"
    assert grid.status in {"converged", "max_level"}
    assert grid.dimension == 2
    assert grid.halfplanes.shape[1] == 3
    values = grid.inner @ grid.halfplanes[:, :2].T
    assert np.max(values - grid.halfplanes[:, 2]) <= 1e-5


def test_scenario_portfolio_and_binding() -> None:
    quotes = _quotes()
    section = _section(quotes)
    frontier = compute_frontier(section, THRESHOLD, quote_set=quotes)
    centroid = _centroid(frontier.inner)
    inside = scenario_check(
        section, THRESHOLD, float(centroid[0]), float(centroid[1]), quote_set=quotes
    )
    assert inside["feasible"] is True
    assert inside["geometry"] == "inside"
    assert inside["method"] == "lp"
    assert isinstance(inside["frontier"], Frontier)
    outside = scenario_check(section, THRESHOLD, 0.99, 0.0, quote_set=quotes)
    assert outside["feasible"] is False
    assert outside["geometry"] == "outside"
    lower, upper = portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes)
    assert lower <= upper
    assert lower == pytest.approx(frontier.lo[0], abs=5e-6)
    assert upper == pytest.approx(frontier.hi[0], abs=5e-6)
    certificate = binding_quotes(section, THRESHOLD, quote_set=quotes, direction=(1.0, 0.0))
    assert not certificate.quote_positions.empty
    assert certificate.point.shape == (2,)


def test_binding_quotes_relaxed_fallback(caplog: pytest.LogCaptureFixture) -> None:
    quotes = _cross_quote_inconsistent(_quotes())
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    certificate = binding_quotes(_section(quotes), THRESHOLD, quote_set=quotes)
    assert "slack_budget" in set(certificate.structural["constraint"])
    assert "relaxed slack-budget" in caplog.text


def test_axis_support_keeps_two_dimensional_cloud() -> None:
    from crashfrontier.frontier import _classify_axis_cloud

    points = np.array(
        [[0.20, 0.01], [0.55, 0.01], [0.20, 0.04], [0.48, 0.03]],
        dtype=float,
    )
    polygon, dimension, status = _classify_axis_cloud(points, 1e-8)
    assert dimension == 2
    assert status == "polygon"
    assert len(polygon) >= 3


def test_wedge_clip_covers_segment_point_and_empty() -> None:
    from crashfrontier.frontier import _clip_to_target_wedge

    segment = _clip_to_target_wedge(
        np.array([[-0.2, 0.0], [0.5, 0.2]], dtype=float),
        0.9,
    )
    assert len(segment) >= 1
    assert np.all(segment[:, 0] >= -1e-8)
    assert np.all(segment[:, 0] <= 1.0 + 1e-8)
    assert np.all(segment[:, 1] >= -1e-8)
    assert np.all(segment[:, 1] <= 0.9 * np.maximum(segment[:, 0], 0.0) + 1e-6)
    assert len(_clip_to_target_wedge(np.array([[1.4, -0.2]], dtype=float), 0.9)) == 0
    assert len(_clip_to_target_wedge(np.empty((0, 2)), 0.9)) == 0


def test_relaxed_anytime_predicate_uses_raw_transformed_area() -> None:
    from crashfrontier.frontier import _area_predicate
    from crashfrontier.geometry.normalize import BoxNormalizer

    settings = LPSettings(
        anytime_area_absolute_tolerance=1e-15,
        anytime_area_relative_tolerance=0.0,
    )
    box = BoxNormalizer(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    singular = _area_predicate(settings, box, 0.0)
    assert singular(square, square) is False
    finite = _area_predicate(settings, box, 1e-4)
    assert finite(square, square) is True
    tiny = np.array([[0.2, 0.2], [0.3, 0.2], [0.25, 0.3]], dtype=float)
    assert finite(square, tiny) is False


def test_area_outer_keeps_transformed_value() -> None:
    from crashfrontier.frontier import _areas
    from crashfrontier.geometry.depth import transformed_area

    inner = np.array([[0.2, 0.02], [0.4, 0.02], [0.4, 0.04], [0.2, 0.04]], dtype=float)
    outer = np.array([[0.25, 0.025], [0.35, 0.025], [0.35, 0.035], [0.25, 0.035]], dtype=float)
    area, outer_area = _areas(inner, outer, 1e-4)
    expected, _, _ = transformed_area(outer, 1e-4)
    assert outer_area == pytest.approx(expected)
    assert outer_area < area


def test_accumulated_halfplanes_match_outer() -> None:
    from dataclasses import replace

    from crashfrontier.frontier import _apply_accumulated_halfplanes

    quotes = _quotes()
    frontier = compute_frontier(_section(quotes), THRESHOLD, quote_set=quotes)
    square = np.array(
        [[0.20, 0.01], [0.40, 0.01], [0.40, 0.05], [0.20, 0.05]],
        dtype=float,
    )
    frontier = replace(
        frontier,
        inner=square.copy(),
        outer=np.array(
            [[0.10, 0.00], [0.50, 0.00], [0.50, 0.08], [0.10, 0.08]],
            dtype=float,
        ),
        halfplanes=np.array(
            [[1.0, 0.0, 0.40], [-1.0, 0.0, -0.20], [0.0, 1.0, 0.05], [0.0, -1.0, -0.01]],
            dtype=float,
        ),
        lo=np.array([0.10, 0.00]),
        hi=np.array([0.50, 0.20]),
        dimension=2,
        method="hull",
        K=0.9,
        p0=1e-4,
    )
    updated = _apply_accumulated_halfplanes(frontier, [(np.array([1.0, 0.0]), 0.30)])
    assert any(np.isclose(row[0], 1.0) and np.isclose(row[2], 0.30) for row in updated.halfplanes)
    assert np.max(updated.outer[:, 0]) <= 0.30 + 1e-7
    values = np.atleast_2d(updated.outer) @ updated.halfplanes[:, :2].T
    assert np.max(values - updated.halfplanes[:, 2]) <= 1e-6
    for vertex in updated.outer:
        assert updated.contains(float(vertex[0]), float(vertex[1])) != "outside"
    for vertex in updated.inner:
        assert updated.contains(float(vertex[0]), float(vertex[1])) == "inside"


def test_support_cap_recheck_rejects_infeasible_tail() -> None:
    from crashfrontier.frontier import _configure_gamma0

    quotes = pd.DataFrame(
        {
            "quote_id": ["q0", "q1", "q2"],
            "cp_flag": ["P", "P", "C"],
            "strike": [0.5, 0.8, 1.0],
            "bid": [0.01, 0.01, 0.40],
            "ask": [0.50, 0.50, 0.80],
            "x": [0.5, 0.8, 1.0],
            "norm_l": [0.01, 0.01, 0.40],
            "norm_u": [0.50, 0.50, 0.80],
        }
    )
    with pytest.raises(InfeasibleCrossSectionError, match="support_cap=1.2 makes"):
        _configure_gamma0(
            quotes,
            THRESHOLD,
            mode="hard",
            budget=None,
            support_cap=1.2,
            lp_settings=LPSettings(),
            full_strike_count=None,
            warn_relaxed=False,
        )


@pytest.mark.parametrize("entrypoint", ("portfolio", "binding"))
def test_support_cap_public_helpers_preserve_infeasible_metadata(entrypoint: str) -> None:
    quotes = pd.DataFrame(
        {
            "quote_id": ["q0", "q1", "q2"],
            "cp_flag": ["P", "P", "C"],
            "strike": [0.5, 0.8, 1.0],
            "bid": [0.01, 0.01, 0.40],
            "ask": [0.50, 0.50, 0.80],
            "x": [0.5, 0.8, 1.0],
            "norm_l": [0.01, 0.01, 0.40],
            "norm_u": [0.50, 0.50, 0.80],
        }
    )
    section = _section(_quotes())
    with pytest.raises(InfeasibleCrossSectionError) as excinfo:
        if entrypoint == "portfolio":
            portfolio_bound(
                section,
                THRESHOLD,
                1.0,
                0.0,
                quote_set=quotes,
                support_cap=1.2,
            )
        else:
            binding_quotes(
                section,
                THRESHOLD,
                quote_set=quotes,
                support_cap=1.2,
            )
    error = excinfo.value
    assert error.date == section.date
    assert error.expiry == section.expiry
    assert error.quote_set == "custom"
    assert error.gamma == 0
    assert error.support_cap == pytest.approx(1.2)
    assert error.status == "infeasible"


def test_nested_preflight_preserves_infeasible_metadata(monkeypatch) -> None:
    section = _section(_quotes())

    def fail(*_args, **_kwargs):
        raise InfeasibleCrossSectionError("support cap infeasible")

    monkeypatch.setattr("crashfrontier.frontier._configure_gamma0", fail)
    with pytest.raises(InfeasibleCrossSectionError) as excinfo:
        nested_frontiers(
            section,
            THRESHOLD,
            sets=("put_pair", "put_wing"),
            support_cap=1.2,
        )
    error = excinfo.value
    assert error.date == section.date
    assert error.expiry == section.expiry
    assert error.quote_set == "put_wing"
    assert error.gamma == 0
    assert error.support_cap == pytest.approx(1.2)
    assert error.status == "infeasible"


def test_support_cap_rejects_positive_bid_beyond_cap() -> None:
    quotes = _quotes()
    extra = quotes.iloc[[0]].copy()
    extra["x"] = 6.0
    extra["strike"] = 6.0
    extra["norm_l"] = 0.2
    extra["norm_u"] = 0.25
    extra["bid"] = 0.2
    extra["ask"] = 0.25
    extra["quote_id"] = "beyond_cap"
    panel = pd.concat([quotes, extra], ignore_index=True)
    with pytest.raises(InfeasibleCrossSectionError, match="support_cap=5"):
        compute_frontier(_section(panel), THRESHOLD, quote_set=panel, support_cap=5.0)


def test_portfolio_and_binding_parse_support_cap_tokens() -> None:
    quotes = _quotes()
    section = _section(quotes)
    unbounded = portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes, support_cap=None)
    auto = portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes, support_cap="auto")
    none = portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes, support_cap="none")
    assert auto == pytest.approx(unbounded)
    assert none == pytest.approx(unbounded)
    certificate = binding_quotes(section, THRESHOLD, quote_set=quotes, support_cap="none")
    assert np.isfinite(certificate.support_value)
    numeric = portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes, support_cap="5")
    assert np.isfinite(numeric).all()


def test_gamma_and_method_rejected() -> None:
    quotes = _quotes()
    section = _section(quotes)
    with pytest.raises(ValueError, match="unsupported gamma"):
        compute_frontier(section, THRESHOLD, gamma=3, quote_set=quotes)
    with pytest.raises(ValueError, match="unrecognized reconstruction method"):
        compute_frontier(section, THRESHOLD, method="spline", quote_set=quotes)
    with pytest.raises(ValueError, match="gamma=0"):
        portfolio_bound(section, THRESHOLD, 1.0, 0.0, quote_set=quotes, gamma=1)
    with pytest.raises(ValueError, match="gamma=0"):
        binding_quotes(section, THRESHOLD, quote_set=quotes, gamma=1)


def test_default_quote_set_is_put_wing() -> None:
    quotes = _quotes()
    frontier = compute_frontier(_section(quotes), THRESHOLD)
    assert frontier.quote_set == "put_wing"
    assert frontier.n_strikes >= 2
    assert frontier.n_quotes >= 2
    boundary = frontier.depth_boundary(samples_per_edge=8)
    if len(frontier.inner) >= 2:
        assert boundary.ndim == 2


def test_conic_settings_exported_from_config_and_root() -> None:
    from crashfrontier import ConicSettings as RootConicSettings

    assert RootConicSettings is ConicSettings
    settings = ConicSettings()
    assert settings.solver == "CLARABEL"
    assert settings.fallbacks == ("ECOS", "SCS")


@pytest.mark.parametrize("gamma", (1, 2))
def test_compute_frontier_physical_is_two_dimensional(gamma: int) -> None:
    quotes = _quotes()
    section = _section(quotes)
    frontier = compute_frontier(
        section,
        THRESHOLD,
        gamma=gamma,
        quote_set=quotes,
        grid_settings=FAST_GRID,
    )
    assert isinstance(frontier, Frontier)
    assert frontier.gamma == gamma
    assert frontier.mode == "socp"
    assert frontier.method == "grid"
    assert frontier.dimension == 2
    assert frontier.status in {"converged", "max_level"}
    assert frontier.quote_set == "custom"
    assert frontier.support_cap == 5.0
    assert len(frontier.inner) >= 3
    assert len(frontier.outer) >= 3
    for vertex in frontier.inner:
        assert point_in_convex_polygon(vertex, frontier.outer, tolerance=1e-6)
    assert frontier.area_outer + 1e-12 >= frontier.area
    assert frontier.diagnostics["elapsed_seconds"] > 0.0
    assert frontier.diagnostics["support_calls"] >= 8
    assert np.isfinite(frontier.diagnostics["max_violation"])
    summary = frontier.summary()
    assert list(summary) == SUMMARY_KEYS
    assert summary["solver_gap"] == pytest.approx(frontier.diagnostics["max_violation"])
    assert np.isfinite(frontier.depth_lo)
    assert np.isfinite(frontier.depth_hi)
    assert frontier.depth_lo <= frontier.depth_hi + 1e-8
    inner_depth = frontier.inner[:, 1] / frontier.inner[:, 0]
    assert frontier.depth_lo <= float(np.min(inner_depth)) + 2e-5
    assert frontier.depth_hi + 2e-5 >= float(np.max(inner_depth))
    assert frontier.diagnostics["inner_depth_lo"] == pytest.approx(
        float(np.min(inner_depth))
    )
    assert np.all(frontier.outer[:, 0] >= -1e-10)
    assert np.all(frontier.outer[:, 0] <= 1.0 + 1e-10)
    assert np.all(frontier.outer[:, 1] >= -1e-10)
    assert np.all(frontier.outer[:, 1] <= THRESHOLD * frontier.outer[:, 0] + 1e-10)
    planes = frontier.halfplane_table()
    assert list(planes.columns) == ["a_prob", "a_loss", "h"]
    assert len(planes) == len(frontier.halfplanes)
    values = frontier.inner @ frontier.halfplanes[:, :2].T
    assert np.max(values - frontier.halfplanes[:, 2]) <= 5e-3
    nodes, weights = scenarios(THRESHOLD)["bimodal"]
    probability, loss, depth = true_targets(nodes, weights, THRESHOLD, gamma=gamma)
    assert frontier.lo[0] - 1e-4 <= probability <= frontier.hi[0] + 1e-4
    assert frontier.lo[1] - 1e-4 <= loss <= frontier.hi[1] + 1e-4
    if np.isfinite(frontier.depth_lo):
        assert frontier.depth_lo - 2e-4 <= depth <= frontier.depth_hi + 2e-4
    json.dumps(frontier.to_dict(), allow_nan=False)


@pytest.mark.parametrize("gamma", (1, 2))
def test_default_physical_quote_set_is_full_otm(gamma: int) -> None:
    quotes = _quotes()
    frontier = compute_frontier(
        _section(quotes),
        THRESHOLD,
        gamma=gamma,
        grid_settings=FAST_GRID,
    )
    assert frontier.quote_set == "full_otm"
    assert frontier.gamma == gamma
    assert frontier.support_cap == 5.0


@pytest.mark.parametrize("gamma", (1, 2))
def test_nested_physical_outer_envelopes_are_nested(gamma: int, dataset) -> None:
    section = dataset.cross_section("2021-07-21", days=14)
    fronts = nested_frontiers(
        section,
        THRESHOLD,
        gamma=gamma,
        grid_settings=NESTED_GRID,
    )
    assert list(fronts) == list(PHYSICAL_SETS)
    assert all(frontier.gamma == gamma for frontier in fronts.values())
    assert all(frontier.mode == "socp" for frontier in fronts.values())
    assert all(frontier.dimension == 2 for frontier in fronts.values())
    names = list(fronts)
    for index in range(1, len(names)):
        inner_set = fronts[names[index]]
        outer_set = fronts[names[index - 1]]
        for vertex in inner_set.outer:
            assert point_in_convex_polygon(vertex, outer_set.outer, tolerance=5e-6)
        for vertex in inner_set.inner:
            assert point_in_convex_polygon(vertex, outer_set.outer, tolerance=5e-6)
    areas = [fronts[name].area_outer for name in names]
    for left, right in zip(areas, areas[1:]):
        assert left + 1e-8 >= right
    pair = fronts["put_pair"]
    wing = fronts["put_wing"]
    assert np.allclose(wing.normalizer.origin, pair.normalizer.origin)
    assert np.allclose(wing.normalizer.scale, pair.normalizer.scale)


@pytest.mark.parametrize("gamma", (1, 2))
def test_physical_infeasible_full_otm_raises_with_metadata(gamma: int) -> None:
    quotes = _otm_call_conflict(_quotes())
    section = _section(quotes)
    wing = compute_frontier(
        section,
        THRESHOLD,
        gamma=gamma,
        quote_set="put_wing",
        grid_settings=FAST_GRID,
    )
    assert wing.quote_set == "put_wing"
    assert wing.dimension in {1, 2}
    with pytest.raises(InfeasibleCrossSectionError) as raised:
        compute_frontier(
            section,
            THRESHOLD,
            gamma=gamma,
            quote_set="full_otm",
            grid_settings=FAST_GRID,
        )
    error = raised.value
    assert error.quote_set == "full_otm"
    assert error.gamma == gamma
    assert error.support_cap == 5.0
    assert error.date == section.date
    assert error.expiry == section.expiry
    assert error.status
    message = str(error)
    assert "full_otm" in message
    assert "put_wing" in message
    assert "support_cap=None" in message
    with pytest.raises(InfeasibleCrossSectionError) as nested:
        nested_frontiers(
            section,
            THRESHOLD,
            gamma=gamma,
            grid_settings=FAST_GRID,
        )
    nested_error = nested.value
    assert nested_error.quote_set == "full_otm"
    assert "completed:" in str(nested_error)
    assert "put_wing" in str(nested_error)
    assert "failed at quote_set='full_otm'" in str(nested_error)


@pytest.mark.parametrize("gamma", (1, 2))
def test_physical_scenario_check_and_unbounded_cap(gamma: int) -> None:
    quotes = _quotes()
    section = _section(quotes)
    frontier = compute_frontier(
        section,
        THRESHOLD,
        gamma=gamma,
        quote_set=quotes,
        grid_settings=FAST_GRID,
    )
    centroid = _centroid(frontier.inner)
    inside = scenario_check(
        section,
        THRESHOLD,
        float(centroid[0]),
        float(centroid[1]),
        gamma=gamma,
        quote_set=quotes,
        grid_settings=FAST_GRID,
    )
    assert inside["feasible"] is True
    assert inside["geometry"] == "inside"
    assert inside["method"] == "socp"
    outside = scenario_check(
        section,
        THRESHOLD,
        0.99,
        0.0,
        gamma=gamma,
        quote_set=quotes,
        grid_settings=FAST_GRID,
    )
    assert outside["feasible"] is False
    assert outside["geometry"] == "outside"
    unbounded = compute_frontier(
        section,
        THRESHOLD,
        gamma=gamma,
        quote_set=quotes,
        support_cap=None,
        grid_settings=FAST_GRID,
    )
    assert unbounded.support_cap is None
    assert unbounded.dimension in {1, 2}
