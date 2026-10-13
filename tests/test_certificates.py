from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crashfrontier.config import LPSettings
from crashfrontier.riskneutral import DualCertificate, dual_certificate, interior_margin
from crashfrontier.riskneutral.lp import (
    build_problem,
    configure_constraints,
    impose_support_cap,
    support,
)
from tests.synthetic import scenarios, synthetic_quotes

SETTINGS = LPSettings()
THRESHOLD = 0.90
SPREAD = 1e-4
COST_TOLERANCE = 1e-8
GAP_TOLERANCE = 1e-9
RESIDUAL_TOLERANCE = 1e-8


def _quotes(spread: float = SPREAD, scenario_name: str = "bimodal") -> pd.DataFrame:
    nodes, weights = scenarios(THRESHOLD)[scenario_name]
    return synthetic_quotes(nodes, weights, spread=spread)


def _hard_system(spread: float = SPREAD, scenario_name: str = "bimodal"):
    problem = build_problem(_quotes(spread, scenario_name), THRESHOLD)
    system = configure_constraints(problem, SETTINGS, "hard")
    assert system is not None
    return problem, system


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


def _quote_cost(certificate: DualCertificate) -> float:
    if certificate.quote_positions.empty:
        return 0.0
    return float(
        np.dot(
            certificate.quote_positions["weight"].to_numpy(dtype=float),
            certificate.quote_positions["norm_price"].to_numpy(dtype=float),
        )
    )


def _structural_cost(certificate: DualCertificate, system) -> float:
    if certificate.structural.empty:
        return 0.0
    rhs = {
        name: float(value)
        for name, value in zip(system.row_names, np.asarray(system.b_ub, dtype=float))
    }
    total = 0.0
    for row in certificate.structural.itertuples(index=False):
        total += float(row.weight) * rhs[str(row.constraint)]
    return total


def test_interior_margin_is_positive_with_spread() -> None:
    problem, _system = _hard_system()
    success, eta_star, message = interior_margin(problem, SETTINGS)
    assert success
    assert eta_star > SETTINGS.interior_cutoff
    assert eta_star > 1e-8
    assert eta_star <= 0.5
    assert isinstance(message, str) and message


def test_interior_margin_is_zero_when_quotes_have_zero_width() -> None:
    problem, _system = _hard_system(spread=0.0)
    widths = problem.quotes["norm_u"] - problem.quotes["norm_l"]
    assert np.allclose(widths.to_numpy(dtype=float), 0.0)
    success, eta_star, _message = interior_margin(problem, SETTINGS)
    assert success
    assert eta_star == pytest.approx(0.0)


def test_dual_certificate_static_cost_matches_support_value() -> None:
    _problem, system = _hard_system()
    result = support(system, SETTINGS, [1.0, 0.0], lexicographic=False)
    assert result.success
    certificate = dual_certificate(system, result)
    assert certificate.primal_dual_gap < GAP_TOLERANCE
    assert certificate.duality_gap < GAP_TOLERANCE
    assert not certificate.quote_positions.empty
    cost = certificate.cash + certificate.forward + _quote_cost(certificate)
    assert cost == pytest.approx(certificate.support_value, abs=COST_TOLERANCE)
    assert cost == pytest.approx(result.value, abs=COST_TOLERANCE)


def test_scipy_marginals_signs_and_min_problem_gap() -> None:
    problem, system = _hard_system()
    result = support(system, SETTINGS, [1.0, 0.0], lexicographic=False)
    assert result.success
    lambda_min = np.asarray(result.primary.ineqlin.marginals, dtype=float)
    nu_min = np.asarray(result.primary.eqlin.marginals, dtype=float)
    assert np.max(lambda_min) <= 1e-8
    dual_min = float(np.dot(system.b_ub, lambda_min) + np.dot(problem.b_eq, nu_min))
    assert dual_min == pytest.approx(float(result.primary.fun), abs=GAP_TOLERANCE)
    certificate = dual_certificate(system, result)
    scale = -float(result.value) / float(result.primary.fun)
    assert certificate.primal_dual_gap == pytest.approx(
        abs(float(result.primary.fun) - dual_min), abs=1e-15
    )
    assert certificate.cash == pytest.approx(-scale * nu_min[0], abs=1e-12)
    assert certificate.forward == pytest.approx(-scale * nu_min[1], abs=1e-12)
    quotes = certificate.quote_positions
    asks = quotes.loc[quotes["side"].eq("ask")]
    bids = quotes.loc[quotes["side"].eq("bid")]
    if not asks.empty:
        merged = asks.merge(problem.quotes[["quote_id", "ask", "norm_u"]], on="quote_id")
        assert np.allclose(merged["price"].to_numpy(dtype=float), merged["ask"].to_numpy(dtype=float))
        assert np.allclose(
            merged["norm_price"].to_numpy(dtype=float), merged["norm_u"].to_numpy(dtype=float)
        )
    if not bids.empty:
        merged = bids.merge(problem.quotes[["quote_id", "bid", "norm_l"]], on="quote_id")
        assert np.allclose(merged["price"].to_numpy(dtype=float), merged["bid"].to_numpy(dtype=float))
        assert np.allclose(
            merged["norm_price"].to_numpy(dtype=float), merged["norm_l"].to_numpy(dtype=float)
        )
        assert (bids["weight"].to_numpy(dtype=float) <= 1e-8).all()


@pytest.mark.parametrize("direction", ([1.0, 0.0], [0.0, 1.0], [1.0, -0.4], [10.0, 0.0]))
def test_dual_certificate_cost_identity_across_directions(direction: list[float]) -> None:
    _problem, system = _hard_system()
    result = support(system, SETTINGS, direction, lexicographic=False)
    assert result.success
    certificate = dual_certificate(system, result)
    cost = certificate.cash + certificate.forward + _quote_cost(certificate)
    assert cost == pytest.approx(result.value, abs=COST_TOLERANCE)
    assert np.allclose(certificate.direction, np.asarray(direction, dtype=float), atol=1e-8)
    assert certificate.max_stationarity_residual < RESIDUAL_TOLERANCE
    assert certificate.max_complementarity_residual < RESIDUAL_TOLERANCE


def test_dual_certificate_quote_and_structural_columns() -> None:
    _problem, system = _hard_system()
    result = support(system, SETTINGS, [1.0, 0.0], lexicographic=True)
    certificate = dual_certificate(system, result)
    quotes = certificate.quote_positions
    assert list(quotes.columns) == [
        "quote_id",
        "cp_flag",
        "strike",
        "side",
        "weight",
        "price",
        "norm_price",
    ]
    assert quotes["side"].isin(["bid", "ask"]).all()
    assert quotes["cp_flag"].eq("P").all()
    structural = certificate.structural
    assert list(structural.columns) == ["constraint", "weight"]
    assert structural["constraint"].str.match(
        r"^(cell_|threshold_|slack_budget|support_cap)"
    ).all()
    assert 0.0 <= certificate.threshold_atom <= 1.0 + 1e-8
    assert certificate.point.shape == (2,)
    frame = certificate.to_frame()
    assert list(frame.columns) == ["kind", "label", "weight"]
    assert set(frame["kind"]) >= {"cash", "forward", "quote", "threshold_atom"}
    assert frame.loc[frame["kind"].eq("cash"), "weight"].iloc[0] == pytest.approx(
        certificate.cash
    )
    assert frame.loc[frame["kind"].eq("forward"), "weight"].iloc[0] == pytest.approx(
        certificate.forward
    )
    assert frame.loc[frame["kind"].eq("threshold_atom"), "weight"].iloc[0] == pytest.approx(
        certificate.threshold_atom
    )


def test_relaxed_certificate_includes_slack_budget_row() -> None:
    quotes = _cross_quote_inconsistent(_quotes())
    problem = build_problem(quotes, THRESHOLD)
    system = configure_constraints(problem, SETTINGS, "relaxed")
    assert system is not None
    assert system.row_names[-1] == "slack_budget"
    result = support(system, SETTINGS, [1.0, 0.0], lexicographic=False)
    assert result.success
    certificate = dual_certificate(system, result)
    assert "slack_budget" in set(certificate.structural["constraint"])
    assert len(result.primary.ineqlin.marginals) == len(system.row_names)
    assert len(result.primary.ineqlin.marginals) == system.A_ub.shape[0]
    quote_only = certificate.cash + certificate.forward + _quote_cost(certificate)
    full = quote_only + _structural_cost(certificate, system)
    assert full == pytest.approx(result.value, abs=COST_TOLERANCE)
    budget_weight = float(
        certificate.structural.loc[
            certificate.structural["constraint"].eq("slack_budget"), "weight"
        ].iloc[0]
    )
    assert quote_only == pytest.approx(
        result.value - budget_weight * float(system.budget), abs=1e-7
    )


def test_support_cap_row_is_compatible_with_dual_certificate() -> None:
    _problem, system = _hard_system()
    capped = impose_support_cap(system, 5.0)
    assert capped.row_names[-1] == "support_cap"
    result = support(capped, SETTINGS, [1.0, 1.0], lexicographic=False)
    assert result.success
    certificate = dual_certificate(capped, result)
    assert len(result.primary.ineqlin.marginals) == len(capped.row_names)
    cost = (
        certificate.cash
        + certificate.forward
        + _quote_cost(certificate)
        + _structural_cost(certificate, capped)
    )
    assert cost == pytest.approx(result.value, abs=COST_TOLERANCE)


def test_dual_certificate_rejects_failed_support() -> None:
    _problem, system = _hard_system()
    result = support(system, SETTINGS, [1.0, 0.0], lexicographic=False)
    failed = result.__class__(
        success=False,
        point=result.point,
        primary_point=result.primary_point,
        value=result.value,
        upper_value=result.upper_value,
        duality_gap=result.duality_gap,
        x=None,
        primary=result.primary,
        secondary=None,
        face_shortfall=result.face_shortfall,
        message="failed",
    )
    with pytest.raises(ValueError, match="not usable"):
        dual_certificate(system, failed)
