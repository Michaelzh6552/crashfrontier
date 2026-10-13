from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from crashfrontier.errors import SolverFailure
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.physical import Gamma1Model
from crashfrontier.physical.gamma2 import ConicPoint
from crashfrontier.physical.solve import ConicOutcome, ConicSettings, max_constraint_violation
from tests.synthetic import scenarios, synthetic_quotes, true_targets

THRESHOLD = 0.90
SPREAD = 1e-6
SUPPORT_CAP = 6.0
SCENARIO_NAMES = ("bimodal", "far_tail")
TARGET_TOLERANCE = 2e-6
DEPTH_TOLERANCE = 2e-5
VIOLATION_TOLERANCE = 1e-6


def _quotes(scenario_name: str, threshold: float = THRESHOLD, spread: float = SPREAD) -> pd.DataFrame:
    nodes, weights = scenarios(threshold)[scenario_name]
    return synthetic_quotes(nodes, weights, spread=spread)


def _model(
    scenario_name: str,
    *,
    support_cap: float | None = SUPPORT_CAP,
    spread: float = SPREAD,
    threshold: float = THRESHOLD,
) -> Gamma1Model:
    return Gamma1Model(
        _quotes(scenario_name, threshold, spread),
        threshold,
        support_cap=support_cap,
    )


def _truth(scenario_name: str, threshold: float = THRESHOLD) -> tuple[float, float, float]:
    nodes, weights = scenarios(threshold)[scenario_name]
    return true_targets(nodes, weights, threshold, gamma=1)


def test_finite_cap_sets_last_right_endpoint_and_threshold_cell() -> None:
    quotes = pd.DataFrame(
        {
            "x": [0.80, 1.00],
            "norm_l": [0.20, 0.00],
            "norm_u": [0.22, 0.02],
        }
    )
    capped = Gamma1Model(quotes, 0.9, support_cap=5.0)
    assert np.allclose(capped.boundaries, [0.0, 0.8, 0.9, 1.0, 5.0])
    assert capped.n_cells == 4
    assert np.allclose(capped.cell_starts, [0.0, 0.8, 0.9, 1.0])
    assert capped.threshold_cell == 2
    assert capped._right_endpoint(3) == pytest.approx(5.0)
    unbounded = Gamma1Model(quotes, 0.9, support_cap=None)
    assert np.allclose(unbounded.boundaries, [0.0, 0.8, 0.9, 1.0])
    assert unbounded.n_cells == 4
    assert unbounded._right_endpoint(3) is None
    assert unbounded.threshold_cell == 2


def test_support_cap_drops_zero_bid_outside_and_rejects_positive_bid() -> None:
    quotes = pd.DataFrame(
        {
            "x": [0.80, 1.00, 6.00],
            "norm_l": [0.20, 0.00, 0.00],
            "norm_u": [0.22, 0.02, 0.01],
        }
    )
    model = Gamma1Model(quotes, 0.9, support_cap=5.0)
    assert model.quotes["x"].max() < 5.0
    positive = quotes.copy()
    positive.loc[2, "norm_l"] = 1e-4
    with pytest.raises(ValueError, match="positive call bid"):
        Gamma1Model(positive, 0.9, support_cap=5.0)


def test_constructor_rejects_bad_threshold_and_cap() -> None:
    quotes = _quotes("bimodal")
    with pytest.raises(ValueError, match="strictly between"):
        Gamma1Model(quotes, 0.0)
    with pytest.raises(ValueError, match="support_cap"):
        Gamma1Model(quotes, 0.9, support_cap=1.0)
    with pytest.raises(ValueError, match="quote columns missing"):
        Gamma1Model(quotes.drop(columns=["x"]), 0.9)


def test_below_threshold_cells_use_second_moment_soc() -> None:
    model = _model("bimodal")
    cones = [constraint for constraint in model.constraints if isinstance(constraint, cp.SOC)]
    assert len(cones) == model.threshold_cell
    assert model.v.size == model.threshold_cell


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_feasibility_ok_on_synthetic_quotes(scenario_name: str) -> None:
    result = _model(scenario_name).feasibility()
    assert result.ok
    assert result.violation <= VIOLATION_TOLERANCE
    assert np.all(np.isfinite(result.point))


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_prob_bounds_contain_truth(scenario_name: str) -> None:
    probability, _loss, _depth = _truth(scenario_name)
    lo, hi, violation = _model(scenario_name).bounds("prob")
    assert violation <= VIOLATION_TOLERANCE
    assert lo <= probability + TARGET_TOLERANCE
    assert hi >= probability - TARGET_TOLERANCE
    assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_loss_bounds_contain_truth(scenario_name: str) -> None:
    _probability, loss, _depth = _truth(scenario_name)
    model = _model(scenario_name)
    lo, hi, violation = model.bounds("loss")
    direction = np.asarray(model.direction.value, dtype=float)
    assert direction[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(float(direction[1])) == pytest.approx(1.0)
    assert violation <= VIOLATION_TOLERANCE
    assert lo <= loss + TARGET_TOLERANCE
    assert hi >= loss - TARGET_TOLERANCE
    assert 0.0 <= lo <= hi <= THRESHOLD


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_depth_bounds_contain_truth(scenario_name: str) -> None:
    _probability, _loss, depth = _truth(scenario_name)
    lo, hi, violation, ok = _model(scenario_name).depth_bounds()
    assert ok
    assert violation <= VIOLATION_TOLERANCE
    assert lo <= depth + DEPTH_TOLERANCE
    assert hi >= depth - DEPTH_TOLERANCE
    assert 0.0 <= lo <= hi <= THRESHOLD


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_scenario_feasible_accepts_truth(scenario_name: str) -> None:
    probability, loss, _depth = _truth(scenario_name)
    model = _model(scenario_name)
    assert model.scenario_feasible(probability, loss)
    assert not model.scenario_feasible(1.0, THRESHOLD)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_unbounded_cap_is_feasible_and_last_cell_open(scenario_name: str) -> None:
    model = _model(scenario_name, support_cap=None)
    assert model._right_endpoint(model.n_cells - 1) is None
    result = model.feasibility()
    assert result.ok
    lo, hi, violation = model.bounds("prob")
    assert np.isfinite([lo, hi]).all()
    assert violation <= VIOLATION_TOLERANCE
    assert 0.0 <= lo <= hi <= 1.0


def test_support_cap_tightens_prob_interval() -> None:
    unbounded = _model("bimodal", support_cap=None).bounds("prob")
    compact = _model("bimodal", support_cap=SUPPORT_CAP).bounds("prob")
    assert max(unbounded[2], compact[2]) <= VIOLATION_TOLERANCE
    assert compact[0] >= unbounded[0] - TARGET_TOLERANCE
    assert compact[1] <= unbounded[1] + TARGET_TOLERANCE


def test_threshold_atom_enters_probability_and_cancels_from_loss() -> None:
    probability, loss, depth = _truth("atom_at_k")
    assert loss == pytest.approx(0.0, abs=1e-12)
    assert depth == pytest.approx(0.0, abs=1e-12)
    model = _model("atom_at_k")
    feasible = model.feasibility()
    assert feasible.ok
    prob_lo, prob_hi, prob_violation = model.bounds("prob")
    loss_lo, loss_hi, loss_violation = model.bounds("loss")
    depth_lo, depth_hi, depth_violation, depth_ok = model.depth_bounds()
    assert max(prob_violation, loss_violation, depth_violation) <= VIOLATION_TOLERANCE
    assert prob_lo <= probability + TARGET_TOLERANCE
    assert probability <= prob_hi + TARGET_TOLERANCE
    assert loss_lo <= loss + TARGET_TOLERANCE
    assert loss <= loss_hi + TARGET_TOLERANCE
    assert depth_ok
    assert depth_lo <= depth + DEPTH_TOLERANCE
    assert depth <= depth_hi + DEPTH_TOLERANCE
    assert model.scenario_feasible(max(probability, 0.0), max(loss, 0.0))


def test_interior_margin_positive_with_spread() -> None:
    ok, margin, violation, status = _model("bimodal", spread=1e-4).interior_margin()
    assert ok
    assert margin > 1e-10
    assert margin <= 0.5 + 1e-8
    assert violation <= VIOLATION_TOLERANCE
    assert status in {cp.OPTIMAL, "optimal_inaccurate_certified"}


def test_interior_margin_zero_at_vanishing_spread() -> None:
    ok, margin, violation, status = _model("bimodal", spread=0.0).interior_margin()
    assert not ok
    assert status == "zero_width_quote"
    assert violation <= VIOLATION_TOLERANCE
    assert margin == pytest.approx(0.0, abs=1e-8)


def test_support_point_matches_objective() -> None:
    model = _model("bimodal")
    direction = np.array([1.0, -0.4], dtype=float)
    result = model.support(direction)
    assert result.ok
    assert result.violation <= VIOLATION_TOLERANCE
    assert direction @ result.point == pytest.approx(result.value, abs=1e-7)


def test_support_scales_large_directions() -> None:
    model = _model("bimodal")
    unit = np.array([1.0, -0.4], dtype=float)
    scaled = 250.0 * unit
    small = model.support(unit)
    large = model.support(scaled)
    assert small.ok and large.ok
    assert large.value == pytest.approx(250.0 * small.value, rel=1e-7, abs=1e-8)
    assert np.allclose(large.point, small.point, atol=1e-7)


def test_oracle_maps_to_unit_box() -> None:
    model = _model("bimodal")
    lo_p, hi_p, _ = model.bounds("prob")
    lo_l, hi_l, _ = model.bounds("loss")
    origin = np.array([lo_p, lo_l], dtype=float)
    scale = np.array([hi_p - lo_p, hi_l - lo_l], dtype=float)
    normalizer = BoxNormalizer(origin, scale)
    query = model.oracle(normalizer)
    unit = np.array([1.0, 0.0], dtype=float)
    point, support_value = query(unit)
    assert np.isfinite(point).all()
    assert np.isfinite(support_value)
    assert point[0] == pytest.approx(1.0, abs=1e-3)
    assert 0.0 - 1e-3 <= point[1] <= 1.0 + 1e-3
    assert support_value == pytest.approx(1.0, abs=1e-3)
    raw = model.support(normalizer.raw_direction(unit))
    assert raw.ok
    assert raw.point[0] == pytest.approx(hi_p, abs=1e-6)


def test_oracle_raises_on_failed_support(monkeypatch) -> None:
    model = _model("bimodal")

    def failed(_direction):
        return ConicPoint(
            False,
            np.full(2, np.nan),
            float("nan"),
            "solver_error",
            "CLARABEL",
            float("inf"),
        )

    monkeypatch.setattr(model, "support", failed)
    query = model.oracle(BoxNormalizer(np.zeros(2), np.ones(2)))
    with pytest.raises(SolverFailure, match="support failed"):
        query(np.array([1.0, 0.0]))


def test_nonfinite_support_direction_is_rejected() -> None:
    model = _model("bimodal")
    with pytest.raises(ValueError, match="finite 2-vector"):
        model.support([np.inf, 0.0])
    with pytest.raises(ValueError, match="unknown target"):
        model.bounds("q")


def _embed_dirac(model: Gamma1Model, nodes: np.ndarray, weights: np.ndarray) -> None:
    mass = np.zeros(model.n_cells)
    first = np.zeros(model.n_cells)
    second = np.zeros(model.threshold_cell)
    atom = 0.0
    for location, probability in zip(nodes, weights):
        cell = int(np.searchsorted(model.cell_starts, float(location), side="right") - 1)
        assert 0 <= cell < model.n_cells
        mass[cell] += probability
        first[cell] += probability * float(location)
        if abs(float(location) - model.K) <= 1e-12:
            atom += probability
            continue
        if cell < model.threshold_cell:
            second[cell] += probability * float(location) ** 2
    model.z.value = mass
    model.m.value = first
    model.v.value = second
    model.atom.value = atom


def test_gamma1_two_point_targets_follow_first_moment_weights() -> None:
    nodes = np.array([0.60, 1.40])
    weights = np.array([0.50, 0.50])
    threshold = 0.90
    below = nodes <= threshold
    probability = float(np.sum(weights[below] * nodes[below]))
    loss = float(np.sum(weights[below] * nodes[below] * (threshold - nodes[below])))
    assert probability == pytest.approx(0.30)
    assert loss == pytest.approx(0.09)
    quotes = synthetic_quotes(
        nodes,
        weights,
        strikes=np.array([0.50, 0.60, 0.90, 1.20, 1.40]),
        spread=1e-8,
    )
    model = Gamma1Model(quotes, threshold, support_cap=5.0)
    _embed_dirac(model, nodes, weights)
    assert max_constraint_violation(model.constraints) < 1e-8
    target = np.asarray(model.target.value, dtype=float)
    assert target[0] == pytest.approx(probability, abs=1e-8)
    assert target[1] == pytest.approx(loss, abs=1e-8)
    assert target[0] < 0.40
    lo, hi, violation = model.bounds("prob")
    assert violation <= VIOLATION_TOLERANCE
    assert lo <= probability + TARGET_TOLERANCE
    assert hi >= probability - TARGET_TOLERANCE
    assert hi < 0.40


def test_scenario_feasible_raises_on_solver_failure(monkeypatch) -> None:
    model = _model("bimodal")

    def failed(*_args, **_kwargs):
        return ConicOutcome(
            False,
            float("nan"),
            "solver_error",
            "CLARABEL",
            float("inf"),
            None,
            None,
        )

    monkeypatch.setattr("crashfrontier.physical.gamma1.solve_with_rescue", failed)
    probability, loss, _depth = _truth("bimodal")
    with pytest.raises(SolverFailure, match="not certified"):
        model.scenario_feasible(probability, loss)


def test_depth_bisection_zero_endpoint_stays_inside_tolerance() -> None:
    model = _model("bimodal")
    depth = cp.Variable(nonneg=True)
    outcome = model._bisect_depth("max", [depth == 0.0, depth <= model.K], depth)
    assert outcome.ok
    assert outcome.status != "bisect_failed"
    assert 0.0 <= outcome.value <= 1e-4


def test_depth_bisection_cap_endpoint_stays_inside_tolerance() -> None:
    model = _model("bimodal")
    model.settings = ConicSettings(violation_tolerance=0.0)
    depth = cp.Variable()
    outcome = model._bisect_depth(
        "min",
        [depth == model.K, depth >= 0.0, depth <= model.K],
        depth,
    )
    assert outcome.ok
    assert outcome.status != "bisect_failed"
    assert model.K - 1e-4 <= outcome.value <= model.K


def test_depth_bisection_keeps_known_feasible_cut(monkeypatch) -> None:
    model = _model("bimodal")
    outcomes = iter(
        [
            ConicOutcome(True, 0.0, cp.OPTIMAL, "CLARABEL", 0.0, 1, 0.0),
            ConicOutcome(True, 0.0, cp.OPTIMAL, "CLARABEL", 0.0, 1, 0.0),
            ConicOutcome(
                False,
                float("nan"),
                "solver_error",
                "CLARABEL",
                float("inf"),
                None,
                None,
            ),
        ]
    )
    monkeypatch.setattr(
        "crashfrontier.physical.gamma1.solve_with_rescue",
        lambda *_args, **_kwargs: next(outcomes),
    )
    outcome = model._bisect_depth("max", [], cp.Constant(float(model.K)))
    assert outcome.ok
    assert outcome.status == "depth_bracket_bound"
    assert outcome.value == pytest.approx(model.K)
