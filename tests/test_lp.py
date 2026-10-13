from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scipy.optimize import linprog

from crashfrontier.config import LPSettings
from crashfrontier.riskneutral.lp import (
    _threshold_cell,
    build_problem,
    configure_constraints,
    impose_support_cap,
    marginal_bounds,
    minimum_slack,
    portfolio_bounds,
    scenario_feasible,
    support,
)
from tests.synthetic import scenarios, synthetic_quotes, true_targets

SCENARIO_NAMES = (
    "atom_at_k",
    "atom_below_k",
    "atom_above_k",
    "far_tail",
    "bimodal",
)
THRESHOLDS = (0.85, 0.90)
SETTINGS = LPSettings()
SPREAD = 1e-6
DELTA_TOLERANCE = 1e-8
TARGET_TOLERANCE = 2e-7
DUALITY_TOLERANCE = 1e-9


def _quotes(threshold: float, scenario_name: str) -> pd.DataFrame:
    nodes, weights = scenarios(threshold)[scenario_name]
    return synthetic_quotes(nodes, weights, spread=SPREAD)


def _cross_quote_inconsistent(quotes: pd.DataFrame) -> pd.DataFrame:
    """Disjoint intervals at one strike; each row still has ``norm_l <= norm_u``."""
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


def _two_strike_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quote_id": ["q0", "q1"],
            "symbol": ["", ""],
            "root": ["", ""],
            "cp_flag": ["P", "P"],
            "strike": [0.8, 1.0],
            "bid": [0.20, 0.00],
            "ask": [0.22, 0.02],
            "volume": [10.0, 10.0],
            "open_interest": [10.0, 10.0],
            "impl_volatility": [np.nan, np.nan],
            "x": [0.8, 1.0],
            "log_moneyness": [np.log(0.8), np.log(1.0)],
            "norm_bid": [0.20, 0.00],
            "norm_ask": [0.22, 0.02],
            "norm_l": [0.20, 0.00],
            "norm_u": [0.22, 0.02],
            "role": ["otm_put", "itm"],
        }
    )


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_hard_is_feasible_on_synthetic_quotes(scenario_name: str, threshold: float) -> None:
    problem = build_problem(_quotes(threshold, scenario_name), threshold)
    system = configure_constraints(problem, SETTINGS, "hard")
    assert system is not None
    assert system.mode == "hard"
    assert system.budget == 0.0
    for index in range(system.problem.slack_lower.start, system.problem.slack_upper.stop):
        assert system.problem.bounds[index] == (0.0, 0.0)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_minimum_slack_is_zero_when_hard_feasible(
    scenario_name: str, threshold: float
) -> None:
    problem = build_problem(_quotes(threshold, scenario_name), threshold)
    result = minimum_slack(problem, SETTINGS)
    assert result.success
    assert result.fun == pytest.approx(0.0, abs=DELTA_TOLERANCE)
    system = configure_constraints(problem, SETTINGS, "hard")
    assert system is not None
    assert system.minimum_slack == pytest.approx(result.fun, abs=1e-12)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_relaxed_recovers_cross_quote_inconsistency(
    scenario_name: str, threshold: float
) -> None:
    quotes = _cross_quote_inconsistent(_quotes(threshold, scenario_name))
    assert (quotes["norm_l"] <= quotes["norm_u"]).all()
    problem = build_problem(quotes, threshold)
    assert len(problem.quotes) == len(quotes)
    hard = configure_constraints(problem, SETTINGS, "hard")
    assert hard is None
    result = minimum_slack(problem, SETTINGS)
    assert result.success
    assert result.fun > DELTA_TOLERANCE
    system = configure_constraints(problem, SETTINGS, "relaxed")
    assert system is not None
    assert system.mode == "relaxed"
    assert system.budget == pytest.approx(result.fun + 1e-10)
    assert system.budget > 0.0
    assert system.minimum_slack == pytest.approx(result.fun, abs=1e-12)
    assert system.row_names[-1] == "slack_budget"
    tight = configure_constraints(problem, SETTINGS, "relaxed", budget=0.0)
    assert tight is None


def test_inverted_row_is_dropped_and_hard_stays_feasible() -> None:
    quotes = _quotes(0.9, "bimodal")
    inverted = quotes.copy()
    inverted.loc[inverted.index[0], "norm_l"] = float(inverted.iloc[0]["norm_u"]) + 0.05
    assert float(inverted.iloc[0]["norm_l"]) > float(inverted.iloc[0]["norm_u"])
    problem = build_problem(inverted, 0.9)
    assert len(problem.quotes) == len(quotes) - 1
    assert (problem.quotes["norm_l"] <= problem.quotes["norm_u"]).all()
    assert configure_constraints(problem, SETTINGS, "hard") is not None


def test_variable_layout_row_names_and_slack_weights() -> None:
    quotes = _two_strike_quotes()
    threshold = 0.9
    problem = build_problem(quotes, threshold, full_strike_count=10)
    n_cells = problem.n_cells
    n_quotes = problem.n_quotes
    assert n_quotes == 2
    assert np.allclose(problem.boundaries, np.array([0.0, 0.8, 0.9, 1.0]))
    assert problem.threshold_cell == 2
    assert n_cells == 4
    assert problem.slack_lower == slice(2 * n_cells, 2 * n_cells + n_quotes)
    assert problem.slack_upper == slice(2 * n_cells + n_quotes, problem.atom_index)
    assert problem.atom_index == 2 * n_cells + 2 * n_quotes
    assert problem.slack_objective.size == problem.atom_index + 1
    assert problem.quote_row_start == 2 * n_cells - 1
    assert problem.row_names[:2] == ("cell_0_moment_lower", "cell_0_moment_upper")
    assert "cell_3_moment_lower" in problem.row_names
    assert "cell_3_moment_upper" not in problem.row_names
    assert problem.row_names[problem.quote_row_start] == "quote_q0_ask"
    assert problem.row_names[problem.quote_row_start + 1] == "quote_q0_bid"
    assert problem.row_names[problem.quote_row_start + 2] == "quote_q1_ask"
    assert problem.row_names[problem.quote_row_start + 3] == "quote_q1_bid"
    assert problem.row_names[-2] == "threshold_atom_mass"
    assert problem.row_names[-1] == "threshold_atom_residual_upper"

    matrix = problem.A_ub.toarray()
    ask0 = matrix[problem.quote_row_start]
    active = problem.boundaries >= 0.8 - 1e-11
    for cell, flag in enumerate(active):
        expected_mass = 1.0 if flag else 0.0
        expected_prob = -0.8 if flag else 0.0
        assert ask0[n_cells + cell] == pytest.approx(expected_mass)
        assert ask0[cell] == pytest.approx(expected_prob)
    assert ask0[problem.slack_upper.start] == pytest.approx(-1.0)
    assert problem.b_ub[problem.quote_row_start] == pytest.approx(0.22)
    assert problem.b_ub[problem.quote_row_start + 1] == pytest.approx(-0.20)

    atom_row = matrix[problem.row_names.index("threshold_atom_mass")]
    assert atom_row[problem.atom_index] == pytest.approx(1.0)
    assert atom_row[problem.threshold_cell] == pytest.approx(-1.0)
    residual = matrix[problem.row_names.index("threshold_atom_residual_upper")]
    assert residual[n_cells + problem.threshold_cell] == pytest.approx(1.0)
    assert residual[problem.threshold_cell] == pytest.approx(-1.0)
    assert residual[problem.atom_index] == pytest.approx(0.1)

    equality = problem.A_eq.toarray()
    assert np.allclose(equality[0, :n_cells], 1.0)
    assert np.allclose(equality[1, n_cells : 2 * n_cells], 1.0)
    assert np.allclose(problem.b_eq, [1.0, 1.0])

    assert np.allclose(problem.p_target[: problem.threshold_cell], 1.0)
    assert problem.p_target[problem.atom_index] == pytest.approx(1.0)
    assert np.allclose(problem.loss_target[: problem.threshold_cell], threshold)
    assert np.allclose(
        problem.loss_target[n_cells : n_cells + problem.threshold_cell], -1.0
    )
    assert problem.loss_target[problem.atom_index] == pytest.approx(0.0)
    weights = problem.slack_objective[problem.slack_lower]
    assert np.allclose(weights, 1.0 / (10.0 * 1.0))
    assert np.allclose(problem.slack_objective[problem.slack_upper], weights)


def test_default_slack_weights_use_this_sets_strike_count() -> None:
    quotes = _two_strike_quotes()
    extra = quotes.iloc[[0]].copy()
    extra["quote_id"] = "q0b"
    extra["norm_l"] = 0.21
    extra["norm_u"] = 0.23
    panel = pd.concat([quotes, extra], ignore_index=True)
    problem = build_problem(panel, 0.9)
    n_strikes = float(panel["strike"].nunique())
    counts = panel.groupby("strike")["strike"].transform("size").to_numpy(dtype=float)
    expected = 1.0 / (n_strikes * counts)
    assert np.allclose(problem.slack_objective[problem.slack_lower], expected)
    assert np.allclose(problem.slack_objective[problem.slack_upper], expected)


def test_threshold_nearest_boundary_is_unique() -> None:
    quotes = _two_strike_quotes()
    near = quotes.iloc[[0]].copy()
    near["x"] = 0.9 + 3e-12
    near["strike"] = float(near["x"].iloc[0])
    near["quote_id"] = "near_k"
    panel = pd.concat([quotes, near], ignore_index=True)
    problem = build_problem(panel, 0.9)
    rounded = float(np.round(0.9, decimals=12))
    assert problem.boundaries[problem.threshold_cell] == pytest.approx(rounded, abs=0.0)
    distances = np.abs(problem.boundaries - rounded)
    assert int(np.argmin(distances)) == problem.threshold_cell
    tied = np.array([0.0, 1.0, 1.0 + 1e-15], dtype=float)
    with pytest.raises(ValueError, match="threshold boundary is not unique"):
        _threshold_cell(tied, 1.0)


def test_support_cap_inserts_boundary_and_drops_far_quotes() -> None:
    quotes = _two_strike_quotes()
    far = quotes.iloc[[1]].copy()
    far["x"] = 6.0
    far["strike"] = 6.0
    far["quote_id"] = "beyond_cap"
    far["norm_l"] = 0.0
    far["norm_u"] = 0.01
    far["bid"] = 0.0
    far["ask"] = 0.01
    panel = pd.concat([quotes, far], ignore_index=True)
    problem = build_problem(panel, 0.9, support_cap=5.0)
    assert 6.0 not in set(np.round(problem.quotes["x"], decimals=8))
    assert np.any(np.isclose(problem.boundaries, 5.0))
    positive = far.copy()
    positive["norm_l"] = 0.2
    positive["norm_u"] = 0.25
    positive["bid"] = 0.2
    positive["ask"] = 0.25
    with pytest.raises(ValueError, match="positive call bid"):
        build_problem(pd.concat([quotes, positive], ignore_index=True), 0.9, support_cap=5.0)


def test_impose_support_cap_appends_named_row() -> None:
    problem = build_problem(_quotes(0.9, "bimodal"), 0.9)
    system = configure_constraints(problem, SETTINGS, "hard")
    assert system is not None
    capped = impose_support_cap(system, 5.0)
    assert capped.row_names[-1] == "support_cap"
    assert capped.mode == "hard"
    tail = problem.n_cells - 1
    row = capped.A_ub.toarray()[-1]
    assert row[problem.n_cells + tail] == pytest.approx(1.0)
    assert row[tail] == pytest.approx(-5.0)
    assert capped.b_ub[-1] == pytest.approx(0.0)


def test_build_problem_rejects_missing_columns_and_single_strike() -> None:
    quotes = _two_strike_quotes().drop(columns=["quote_id"])
    with pytest.raises(ValueError, match="quote columns missing"):
        build_problem(quotes, 0.9)
    with pytest.raises(ValueError, match="at least two distinct quote strikes"):
        build_problem(_two_strike_quotes().iloc[[0]], 0.9)


def _hard_system(threshold: float, scenario_name: str):
    problem = build_problem(_quotes(threshold, scenario_name), threshold)
    system = configure_constraints(problem, SETTINGS, "hard")
    assert system is not None
    return problem, system


def test_linprog_marginals_match_hand_example() -> None:
    """SciPy reports ∂fun/∂b, not textbook λ ≥ 0.

    Hand LP: minimize ``-x - 2y`` subject to ``x + y ≤ 1``, ``x, y ≥ 0``.
    The vertex is ``(0, 1)`` with ``fun = -2``. Raising the right-hand side
    by ε moves the vertex to ``(0, 1+ε)`` and changes ``fun`` by ``-2ε``, so
    ``∂fun/∂b = -2``. The min-problem dual is ``b · marginals = -2 = fun``;
    the matching max-problem dual used by ``support`` is
    ``-(b · marginals) = 2``.
    """
    cost = np.array([-1.0, -2.0])
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([1.0])
    result = linprog(
        cost,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=((0.0, None), (0.0, None)),
        method="highs-ds",
    )
    assert result.success
    assert result.x[0] == pytest.approx(0.0, abs=1e-12)
    assert result.x[1] == pytest.approx(1.0, abs=1e-12)
    assert result.fun == pytest.approx(-2.0)
    assert result.ineqlin.marginals[0] == pytest.approx(-2.0)
    dual_min = float(b_ub[0] * result.ineqlin.marginals[0])
    assert dual_min == pytest.approx(result.fun)
    assert -dual_min == pytest.approx(2.0)

    shifted = linprog(
        cost,
        A_ub=A_ub,
        b_ub=np.array([1.05]),
        bounds=((0.0, None), (0.0, None)),
        method="highs-ds",
    )
    assert shifted.success
    assert shifted.fun == pytest.approx(-2.1)
    assert shifted.fun - result.fun == pytest.approx(0.05 * result.ineqlin.marginals[0])


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_true_targets_lie_in_bounds_and_are_feasible(
    scenario_name: str, threshold: float
) -> None:
    nodes, weights = scenarios(threshold)[scenario_name]
    probability, loss, _depth = true_targets(nodes, weights, threshold, gamma=0)
    _problem, system = _hard_system(threshold, scenario_name)
    lo, hi = marginal_bounds(system, SETTINGS)
    assert lo[0] - TARGET_TOLERANCE <= probability <= hi[0] + TARGET_TOLERANCE
    assert lo[1] - TARGET_TOLERANCE <= loss <= hi[1] + TARGET_TOLERANCE
    assert hi[0] >= lo[0]
    assert hi[1] >= lo[1]
    assert scenario_feasible(system, SETTINGS, probability, loss)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_support_duality_gap_and_portfolio_order(
    scenario_name: str, threshold: float
) -> None:
    _problem, system = _hard_system(threshold, scenario_name)
    for direction in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -0.4)):
        result = support(system, SETTINGS, direction, lexicographic=True)
        assert result.success
        assert result.message == "ok"
        assert result.duality_gap < DUALITY_TOLERANCE
        assert result.upper_value == pytest.approx(result.value + result.duality_gap)
        assert result.upper_value >= result.value
        assert result.face_shortfall >= 0.0
        assert result.face_shortfall < 1e-8
        assert result.x is not None
        vector = np.asarray(direction, dtype=float)
        assert vector @ result.point == pytest.approx(result.value, abs=1e-8)
        assert vector @ result.primary_point == pytest.approx(result.value, abs=1e-8)
    low, high = portfolio_bounds(system, SETTINGS, 1.0, 1.0)
    assert low <= high
    other_low, other_high = portfolio_bounds(system, SETTINGS, 1.0, -0.4)
    assert other_low <= other_high


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_marginal_bounds_use_signed_upper_values(
    scenario_name: str, threshold: float
) -> None:
    _problem, system = _hard_system(threshold, scenario_name)
    lo, hi = marginal_bounds(system, SETTINGS)
    for axis, unit in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]))):
        high = support(system, SETTINGS, unit, lexicographic=False)
        low = support(system, SETTINGS, -unit, lexicographic=False)
        assert high.success and low.success
        assert hi[axis] == pytest.approx(max(high.upper_value, -low.upper_value, 0.0))
        assert lo[axis] == pytest.approx(max(-low.upper_value, 0.0))
        assert lo[axis] <= hi[axis]


def test_support_value_scales_with_direction() -> None:
    _problem, system = _hard_system(0.9, "bimodal")
    base = support(system, SETTINGS, [1.0, -0.4], lexicographic=False)
    scaled = support(system, SETTINGS, [10_000.0, -4_000.0], lexicographic=False)
    assert base.success and scaled.success
    assert scaled.value == pytest.approx(10_000.0 * base.value, rel=1e-9, abs=1e-9)
    assert scaled.duality_gap < DUALITY_TOLERANCE
    assert base.duality_gap < DUALITY_TOLERANCE


def test_support_dual_matches_scipy_marginal_formula() -> None:
    problem, system = _hard_system(0.9, "bimodal")
    direction = np.array([1.0, -0.4], dtype=float)
    scale = max(float(np.max(np.abs(direction))), 1.0)
    result = support(system, SETTINGS, direction, lexicographic=False)
    assert result.success
    dual_min = float(
        np.dot(system.b_ub, result.primary.ineqlin.marginals)
        + np.dot(problem.b_eq, result.primary.eqlin.marginals)
    )
    dual = -dual_min * scale
    assert dual == pytest.approx(result.value, abs=1e-9)
    assert result.primary.fun == pytest.approx(dual_min, abs=1e-9)
    assert result.value == pytest.approx(-result.primary.fun * scale, abs=1e-12)


def test_lexicographic_picks_rotated_endpoint() -> None:
    _problem, system = _hard_system(0.9, "bimodal")
    direction = np.array([1.0, 0.0], dtype=float)
    lex = support(system, SETTINGS, direction, lexicographic=True)
    plain = support(system, SETTINGS, direction, lexicographic=False)
    assert lex.success and plain.success
    assert lex.secondary is not None
    assert lex.value == pytest.approx(plain.value, abs=1e-9)
    assert lex.point[0] == pytest.approx(lex.value, abs=1e-8)
    assert lex.point[1] >= plain.point[1] - 1e-8
    assert lex.face_shortfall < 1e-9


def test_nonfinite_support_direction_is_rejected() -> None:
    _problem, system = _hard_system(0.9, "bimodal")
    with pytest.raises(ValueError, match="finite 2-vector"):
        support(system, SETTINGS, [np.inf, 0.0])
    with pytest.raises(ValueError, match="finite 2-vector"):
        support(system, SETTINGS, [1.0])


def test_scenario_feasible_rejects_impossible_targets() -> None:
    _problem, system = _hard_system(0.9, "bimodal")
    assert not scenario_feasible(system, SETTINGS, 1.0, 1.0)
    assert not scenario_feasible(system, SETTINGS, -0.1, 0.0)


def test_relaxed_support_succeeds_on_inconsistent_quotes() -> None:
    quotes = _cross_quote_inconsistent(_quotes(0.9, "bimodal"))
    problem = build_problem(quotes, 0.9)
    system = configure_constraints(problem, SETTINGS, "relaxed")
    assert system is not None
    assert system.budget > 0.0
    result = support(system, SETTINGS, [1.0, 1.0])
    assert result.success
    assert result.duality_gap < DUALITY_TOLERANCE
    lo, hi = marginal_bounds(system, SETTINGS)
    assert np.all(lo <= hi)
    low, high = portfolio_bounds(system, SETTINGS, 0.5, 0.5)
    assert low <= high
