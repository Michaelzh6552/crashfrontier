from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from crashfrontier.errors import SolverFailure
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.physical.cones import moment_cone_order2, moment_cone_order3, psd2
from crashfrontier.physical.gamma2 import Gamma2Model
from crashfrontier.physical.solve import (
    ConicOutcome,
    ConicSettings,
    max_constraint_violation,
    solve_with_rescue,
    solver_options,
)
from tests.synthetic import scenarios, synthetic_quotes, true_targets

THRESHOLD = 0.90
SPREAD = 1e-6
SUPPORT_CAP = 6.0
SCENARIO_NAMES = ("bimodal", "far_tail")
TARGET_TOLERANCE = 2e-6
DEPTH_TOLERANCE = 2e-5
UNBOUNDED_PROB_LO = 1e-3
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
) -> Gamma2Model:
    return Gamma2Model(
        _quotes(scenario_name, threshold, spread),
        threshold,
        support_cap=support_cap,
    )


def _truth(scenario_name: str, threshold: float = THRESHOLD) -> tuple[float, float, float]:
    nodes, weights = scenarios(threshold)[scenario_name]
    return true_targets(nodes, weights, threshold, gamma=2)


def _feasible(constraints) -> object:
    problem = cp.Problem(cp.Minimize(0), constraints)
    return solve_with_rescue(problem, constraints, ConicSettings(), warm_start=False)


def test_solver_options_match_settings() -> None:
    settings = ConicSettings(tolerance=2e-8, max_iterations=123)
    clarabel = solver_options("CLARABEL", settings)
    assert clarabel == {
        "tol_gap_abs": 2e-8,
        "tol_gap_rel": 2e-8,
        "tol_feas": 2e-8,
        "max_iter": 123,
    }
    ecos = solver_options("ECOS", settings)
    assert ecos == {
        "abstol": 2e-8,
        "reltol": 2e-8,
        "feastol": 2e-8,
        "max_iters": 123,
    }
    scs = solver_options("SCS", settings)
    assert scs["eps"] == pytest.approx(1e-6)
    assert scs["max_iters"] == 20_000
    assert solver_options("UNKNOWN") == {}


def test_max_constraint_violation_uses_soc_residual() -> None:
    feasible = cp.SOC(cp.Constant(5.0), cp.Constant([3.0, 4.0]))
    violated = cp.SOC(cp.Constant(1.0), cp.Constant([3.0, 4.0]))
    assert max_constraint_violation([feasible]) == pytest.approx(0.0)
    assert max_constraint_violation([violated]) == pytest.approx(4.0)
    slack = cp.Variable()
    slack.value = -0.2
    inequality = slack >= 0.0
    assert max_constraint_violation([inequality]) == pytest.approx(0.2)


def test_psd2_accepts_positive_definite_matrix() -> None:
    upper, off, lower = cp.Variable(), cp.Variable(), cp.Variable()
    constraints = [upper == 2.0, off == 1.0, lower == 2.0] + psd2(upper, off, lower)
    outcome = _feasible(constraints)
    assert outcome.ok
    assert outcome.status in {cp.OPTIMAL, "optimal_inaccurate_certified"}
    assert outcome.violation <= VIOLATION_TOLERANCE


def test_psd2_rejects_indefinite_matrix() -> None:
    upper, off, lower = cp.Variable(), cp.Variable(), cp.Variable()
    constraints = [upper == 1.0, off == 2.0, lower == 1.0] + psd2(upper, off, lower)
    outcome = _feasible(constraints)
    assert not outcome.ok


def test_moment_cone_order3_contains_dirac_on_bounded_cell() -> None:
    left, right, location = 0.5, 0.9, 0.8
    moments = [cp.Variable() for _ in range(4)]
    constraints = [
        moments[power] == location**power for power in range(4)
    ] + moment_cone_order3(*moments, left, right)
    outcome = _feasible(constraints)
    assert outcome.ok
    assert outcome.violation <= VIOLATION_TOLERANCE


def test_moment_cone_order3_unbounded_contains_far_atom() -> None:
    left, location = 1.0, 4.0
    moments = [cp.Variable() for _ in range(4)]
    constraints = [
        moments[power] == location**power for power in range(4)
    ] + moment_cone_order3(*moments, left, None)
    outcome = _feasible(constraints)
    assert outcome.ok


def test_moment_cone_order2_rejects_mean_outside_cell() -> None:
    y0, y1, y2 = cp.Variable(), cp.Variable(), cp.Variable()
    constraints = [y0 == 1.0, y1 == 0.2, y2 == 0.04] + moment_cone_order2(
        y0, y1, y2, 0.5, 0.9
    )
    outcome = _feasible(constraints)
    assert not outcome.ok


def test_solve_with_rescue_returns_optimal_on_socp() -> None:
    scale = cp.Variable(nonneg=True)
    constraints = psd2(scale, 1.0, scale) + [scale >= 1.0]
    problem = cp.Problem(cp.Minimize(scale), constraints)
    outcome = solve_with_rescue(problem, constraints, ConicSettings(), warm_start=False)
    assert outcome.ok
    assert outcome.status == cp.OPTIMAL
    assert outcome.value == pytest.approx(1.0, abs=1e-7)
    assert outcome.violation <= VIOLATION_TOLERANCE
    assert outcome.solver == "CLARABEL"
    assert outcome.solve_time is not None


def test_solve_with_rescue_cold_start_after_inaccurate(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeProblem:
        status = cp.OPTIMAL_INACCURATE
        value = 1.0
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del verbose
            calls.append((str(solver), bool(warm_start)))
            if len(calls) == 1:
                self.status = cp.OPTIMAL_INACCURATE
                self.value = 1.0
                return
            self.status = cp.OPTIMAL
            self.value = 1.0

    outcome = solve_with_rescue(FakeProblem(), [], ConicSettings(), warm_start=True)
    assert calls == [("CLARABEL", True), ("CLARABEL", False)]
    assert outcome.ok
    assert outcome.status == cp.OPTIMAL
    assert outcome.value == pytest.approx(1.0)


def test_solve_with_rescue_certifies_agreeing_inaccurate() -> None:
    class FakeProblem:
        status = cp.OPTIMAL_INACCURATE
        value = 0.25
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del solver, warm_start, verbose
            self.status = cp.OPTIMAL_INACCURATE
            self.value = 0.25

    outcome = solve_with_rescue(
        FakeProblem(), [], ConicSettings(fallbacks=()), warm_start=True
    )
    assert outcome.ok
    assert outcome.status == "optimal_inaccurate_certified"
    assert outcome.solver == "CLARABEL|CLARABEL"
    assert outcome.violation == pytest.approx(0.0)


def test_solve_with_rescue_uncertified_when_inaccurate_values_disagree() -> None:
    class FakeProblem:
        status = cp.OPTIMAL_INACCURATE
        value = 1.0
        solver_stats = None
        calls = 0

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del solver, warm_start, verbose
            self.calls += 1
            self.status = cp.OPTIMAL_INACCURATE
            self.value = 1.0 if self.calls == 1 else 2.0

    outcome = solve_with_rescue(
        FakeProblem(), [], ConicSettings(fallbacks=()), warm_start=True
    )
    assert not outcome.ok
    assert outcome.status == "uncertified_optimal_inaccurate"
    assert np.isnan(outcome.value)


def test_solve_with_rescue_uses_fallback_optimal(monkeypatch) -> None:
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "ECOS"])

    class Recorder:
        status = cp.OPTIMAL_INACCURATE
        value = 1.0
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del verbose
            if str(solver) == "ECOS":
                self.status = cp.OPTIMAL
                self.value = 0.4
                return
            self.status = cp.OPTIMAL_INACCURATE
            self.value = 1.0 if warm_start else 2.0

    outcome = solve_with_rescue(
        Recorder(), [], ConicSettings(fallbacks=("ECOS", "SCS")), warm_start=True
    )
    assert outcome.ok
    assert outcome.status == cp.OPTIMAL
    assert outcome.solver == "ECOS"
    assert outcome.value == pytest.approx(0.4)


def test_solve_with_rescue_treats_solver_error_as_failure() -> None:
    class FakeProblem:
        status = "solver_error"
        value = None
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del solver, warm_start, verbose
            raise cp.error.SolverError("unavailable")

    outcome = solve_with_rescue(FakeProblem(), [], ConicSettings(), warm_start=False)
    assert not outcome.ok
    assert outcome.status == "solver_error"
    assert outcome.violation == float("inf")


def test_depth_bisection_does_not_treat_solver_error_as_infeasible(monkeypatch) -> None:
    outcomes = iter(
        [
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
        "crashfrontier.physical.gamma2.solve_with_rescue",
        lambda *_args, **_kwargs: next(outcomes),
    )
    outcome = _model("bimodal")._bisect_depth("max", [], cp.Constant(0.0))
    assert outcome.ok
    assert outcome.status == "depth_bracket_bound"
    assert outcome.value == pytest.approx(THRESHOLD)


def test_finite_cap_sets_last_right_endpoint_and_threshold_cell() -> None:
    quotes = pd.DataFrame(
        {
            "x": [0.80, 1.00],
            "norm_l": [0.20, 0.00],
            "norm_u": [0.22, 0.02],
        }
    )
    capped = Gamma2Model(quotes, 0.9, support_cap=5.0)
    assert np.allclose(capped.boundaries, [0.0, 0.8, 0.9, 1.0, 5.0])
    assert capped.n_cells == 4
    assert np.allclose(capped.cell_starts, [0.0, 0.8, 0.9, 1.0])
    assert capped.threshold_cell == 2
    assert capped._right_endpoint(3) == pytest.approx(5.0)
    unbounded = Gamma2Model(quotes, 0.9, support_cap=None)
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
    model = Gamma2Model(quotes, 0.9, support_cap=5.0)
    assert model.quotes["x"].max() < 5.0
    positive = quotes.copy()
    positive.loc[2, "norm_l"] = 1e-4
    with pytest.raises(ValueError, match="positive call bid"):
        Gamma2Model(positive, 0.9, support_cap=5.0)


def test_constructor_rejects_bad_threshold_and_cap() -> None:
    quotes = _quotes("bimodal")
    with pytest.raises(ValueError, match="strictly between"):
        Gamma2Model(quotes, 0.0)
    with pytest.raises(ValueError, match="support_cap"):
        Gamma2Model(quotes, 0.9, support_cap=1.0)
    with pytest.raises(ValueError, match="quote columns missing"):
        Gamma2Model(quotes.drop(columns=["x"]), 0.9)


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
    if not ok:
        assert np.isnan(lo)
        assert np.isnan(hi)
        assert violation == float("inf")
        return
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
def test_unbounded_cap_sends_prob_lo_to_zero(scenario_name: str) -> None:
    lo, hi, violation = _model(scenario_name, support_cap=None).bounds("prob")
    assert np.isfinite(hi)
    assert violation <= VIOLATION_TOLERANCE
    assert lo < UNBOUNDED_PROB_LO


def test_threshold_atom_enters_probability_and_cancels_from_loss() -> None:
    probability, loss, depth = _truth("atom_at_k")
    model = _model("atom_at_k")
    feasible = model.feasibility()
    assert feasible.ok
    prob_lo, prob_hi, prob_violation = model.bounds("prob")
    loss_lo, loss_hi, loss_violation = model.bounds("loss")
    depth_lo, depth_hi, depth_violation, depth_ok = model.depth_bounds()
    assert max(prob_violation, loss_violation) <= VIOLATION_TOLERANCE
    assert prob_lo <= probability + TARGET_TOLERANCE
    assert probability <= prob_hi + TARGET_TOLERANCE
    assert loss_lo <= loss + TARGET_TOLERANCE
    assert loss <= loss_hi + TARGET_TOLERANCE
    if depth_ok:
        assert depth_violation <= VIOLATION_TOLERANCE
        assert depth_lo <= depth + DEPTH_TOLERANCE
        assert depth <= depth_hi + DEPTH_TOLERANCE
    else:
        assert np.isnan(depth_lo)
        assert np.isnan(depth_hi)
        assert depth_violation > VIOLATION_TOLERANCE
    assert model.scenario_feasible(max(probability, 0.0), max(loss, 0.0))


def test_interior_margin_positive_with_spread() -> None:
    ok, margin, violation, status = _model("bimodal", spread=1e-4).interior_margin()
    assert ok
    assert margin > 1e-10
    assert violation <= VIOLATION_TOLERANCE
    assert status in {cp.OPTIMAL, "optimal_inaccurate_certified"}


def test_interior_margin_zero_at_vanishing_spread() -> None:
    ok, margin, violation, _status = _model("bimodal", spread=0.0).interior_margin()
    assert not ok
    assert violation <= VIOLATION_TOLERANCE
    assert margin == pytest.approx(0.0, abs=1e-8)


def test_support_point_matches_objective() -> None:
    model = _model("bimodal")
    direction = np.array([1.0, -0.4], dtype=float)
    result = model.support(direction)
    assert result.ok
    assert result.violation <= VIOLATION_TOLERANCE
    assert direction @ result.point == pytest.approx(result.value, abs=1e-7)


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
    from crashfrontier.physical.gamma2 import ConicPoint

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


def test_infeasible_inaccurate_is_not_terminal(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeProblem:
        status = cp.INFEASIBLE_INACCURATE
        value = None
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del verbose
            calls.append((str(solver), bool(warm_start)))
            if str(solver) == "ECOS":
                self.status = cp.OPTIMAL
                self.value = 0.2
                return
            self.status = cp.INFEASIBLE_INACCURATE
            self.value = None

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "ECOS"])
    outcome = solve_with_rescue(
        FakeProblem(),
        [],
        ConicSettings(fallbacks=("ECOS",)),
        warm_start=True,
    )
    assert calls[0] == ("CLARABEL", True)
    assert ("CLARABEL", False) in calls
    assert outcome.ok
    assert outcome.status == cp.OPTIMAL
    assert outcome.solver == "ECOS"
    assert outcome.value == pytest.approx(0.2)


def test_two_solvers_cross_confirm_infeasible_inaccurate(monkeypatch) -> None:
    class FakeProblem:
        status = cp.INFEASIBLE_INACCURATE
        value = None
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del solver, warm_start, verbose
            self.status = cp.INFEASIBLE_INACCURATE
            self.value = None

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "ECOS"])
    outcome = solve_with_rescue(
        FakeProblem(),
        [],
        ConicSettings(fallbacks=("ECOS",)),
        warm_start=True,
    )
    assert not outcome.ok
    assert outcome.status == cp.INFEASIBLE_INACCURATE
    assert outcome.solver == "CLARABEL|ECOS"


def test_definitive_fallback_infeasible_survives_primary_solver_error(monkeypatch) -> None:
    class FakeProblem:
        status = None
        value = None
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            del warm_start, verbose
            if str(solver) == "CLARABEL":
                raise cp.SolverError("primary failed")
            self.status = cp.INFEASIBLE
            self.value = None

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "ECOS"])
    outcome = solve_with_rescue(
        FakeProblem(),
        [],
        ConicSettings(fallbacks=("ECOS",)),
        warm_start=True,
    )
    assert not outcome.ok
    assert outcome.status == cp.INFEASIBLE
    assert outcome.solver == "ECOS"


def test_definitive_infeasible_beats_disagreeing_inaccurate_optima(monkeypatch) -> None:
    calls = 0

    class FakeProblem:
        status = None
        value = None
        solver_stats = None

        def solve(self, solver, warm_start, verbose=False, **_kwargs) -> None:
            nonlocal calls
            del warm_start, verbose
            calls += 1
            if str(solver) == "ECOS":
                self.status = cp.INFEASIBLE
                self.value = None
            else:
                self.status = cp.OPTIMAL_INACCURATE
                self.value = float(calls)

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "ECOS"])
    outcome = solve_with_rescue(
        FakeProblem(),
        [],
        ConicSettings(fallbacks=("ECOS",)),
        warm_start=True,
    )
    assert not outcome.ok
    assert outcome.status == cp.INFEASIBLE
    assert outcome.solver == "ECOS"


def _embed_scaled(model: Gamma2Model, nodes: np.ndarray, weights: np.ndarray) -> float:
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    second_moment = float(np.sum(weights * nodes**2))
    scale = 1.0 / second_moment
    mass = np.zeros(model.n_cells)
    first = np.zeros(model.n_cells)
    second = np.zeros(model.n_cells)
    third = np.zeros(model.n_cells)
    atom = 0.0
    for location, probability in zip(nodes, weights):
        weight = float(probability) * scale
        cell = int(np.searchsorted(model.cell_starts, float(location), side="right") - 1)
        assert 0 <= cell < model.n_cells
        mass[cell] += weight
        first[cell] += weight * float(location)
        second[cell] += weight * float(location) ** 2
        if abs(float(location) - model.K) <= 1e-12:
            atom += weight
        elif cell < model.threshold_cell:
            third[cell] += weight * float(location) ** 3
    model.y.value = np.vstack([mass, first, second, third])
    model.t.value = scale
    model.atom.value = atom
    return second_moment


def test_gamma2_two_point_targets_and_m2_follow_cell_formulas() -> None:
    nodes = np.array([0.60, 1.40])
    weights = np.array([0.50, 0.50])
    threshold = 0.90
    below = nodes <= threshold
    second_moment = float(np.sum(weights * nodes**2))
    area2 = float(np.sum(weights[below] * nodes[below] ** 2))
    area3 = float(np.sum(weights[below] * nodes[below] ** 3))
    probability = area2 / second_moment
    loss = (threshold * area2 - area3) / second_moment
    assert second_moment == pytest.approx(1.16)
    assert probability == pytest.approx(0.15517241379310345, rel=0.0, abs=1e-12)
    assert loss == pytest.approx(0.04655172413793104, rel=0.0, abs=1e-12)
    quotes = synthetic_quotes(
        nodes,
        weights,
        strikes=np.array([0.50, 0.60, 0.90, 1.20, 1.40]),
        spread=1e-8,
    )
    model = Gamma2Model(quotes, threshold, support_cap=5.0)
    embedded = _embed_scaled(model, nodes, weights)
    assert embedded == pytest.approx(second_moment)
    assert max_constraint_violation(model.constraints) < 1e-8
    target = np.asarray(model.target.value, dtype=float)
    assert target[0] == pytest.approx(probability, abs=1e-8)
    assert target[1] == pytest.approx(loss, abs=1e-8)
    assert target[0] < 0.20
    lower, upper, violation = model.m2_bounds()
    assert violation <= VIOLATION_TOLERANCE
    assert lower <= second_moment + 2e-5
    assert upper >= second_moment - 2e-5


def test_unbounded_m2_upper_reports_homogeneous_closure_as_infinite() -> None:
    lower, upper, violation = _model("bimodal", support_cap=None).m2_bounds()
    assert np.isfinite(lower)
    assert np.isinf(upper)
    assert violation <= VIOLATION_TOLERANCE


def test_regularity_witness_certifies_both_sides() -> None:
    nodes = np.array([0.60, 1.40])
    weights = np.array([0.50, 0.50])
    strikes = np.array([0.50, 0.60, 0.90, 1.20, 1.40])
    quotes = synthetic_quotes(nodes, weights, strikes=strikes, spread=0.05)
    model = Gamma2Model(quotes, 0.90, support_cap=2.0)
    witness = model.regularity_sufficient_witness()
    assert witness.success
    assert witness.status == "certified"
    assert witness.quote_slack > 0.0
    assert witness.left_mass > 0.0
    assert witness.right_mass > 0.0
    forbidden = np.r_[0.0, 1.0, model.K, model.support_cap, model.quotes["x"].to_numpy()]
    positive = witness.atom_locations[witness.atom_masses > 1e-10]
    assert np.any(positive < 1.0)
    assert np.any(positive > 1.0)
    assert all(not np.isclose(atom, forbidden, atol=1e-12, rtol=0.0).any() for atom in positive)
    ok, margin, violation, _status = model.interior_margin()
    assert ok
    assert margin > 1e-10
    assert violation <= VIOLATION_TOLERANCE


def test_regularity_witness_zero_spread_is_inconclusive() -> None:
    nodes = np.array([0.60, 1.40])
    weights = np.array([0.50, 0.50])
    quotes = synthetic_quotes(
        nodes,
        weights,
        strikes=np.array([0.50, 0.60, 0.90, 1.20, 1.40]),
        spread=0.0,
    )
    witness = Gamma2Model(quotes, 0.90, support_cap=2.0).regularity_sufficient_witness()
    assert not witness.success
    assert witness.status.startswith("inconclusive:")


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

    monkeypatch.setattr("crashfrontier.physical.gamma2.solve_with_rescue", failed)
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


def test_threshold_atom_depth_upper_bound_contains_feasible_cut() -> None:
    nodes = np.array([THRESHOLD, 1.0, 1.10])
    weights = np.array([0.15, 0.70, 0.15])
    strikes = np.unique(
        np.concatenate(
            [np.linspace(0.25, 1.8, 16), nodes, np.array([THRESHOLD, 1.0])]
        )
    )
    quotes = synthetic_quotes(nodes, weights, strikes=strikes, spread=SPREAD)
    model = Gamma2Model(
        quotes,
        THRESHOLD,
        support_cap=SUPPORT_CAP,
        settings=ConicSettings(max_iterations=2000),
    )
    _low, high, violation, ok = model.depth_bounds()
    assert ok
    assert violation <= VIOLATION_TOLERANCE
    assert high >= 0.30 - DEPTH_TOLERANCE


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
        "crashfrontier.physical.gamma2.solve_with_rescue",
        lambda *_args, **_kwargs: next(outcomes),
    )
    outcome = model._bisect_depth("max", [], cp.Constant(float(model.K)))
    assert outcome.ok
    assert outcome.status == "depth_bracket_bound"
    assert outcome.value == pytest.approx(model.K)


def test_depth_bisection_ignores_unconfirmed_infeasible(monkeypatch) -> None:
    monkeypatch.setattr(
        "crashfrontier.physical.gamma2.solve_with_rescue",
        lambda *_args, **_kwargs: ConicOutcome(
            False,
            float("nan"),
            "uncertified_infeasible_inaccurate",
            "CLARABEL",
            float("inf"),
            None,
            None,
        ),
    )
    outcome = _model("bimodal")._bisect_depth("max", [], cp.Constant(0.0))
    assert not outcome.ok
    assert outcome.status == "uncertified_infeasible_inaccurate"
    assert np.isnan(outcome.value)


def test_depth_bisection_does_not_reject_uncertified_optimum(monkeypatch) -> None:
    monkeypatch.setattr(
        "crashfrontier.physical.gamma2.solve_with_rescue",
        lambda *_args, **_kwargs: ConicOutcome(
            False,
            float("nan"),
            "uncertified_optimal_inaccurate",
            "CLARABEL",
            float("inf"),
            None,
            None,
        ),
    )
    outcome = _model("bimodal")._bisect_depth("max", [], cp.Constant(0.0))
    assert not outcome.ok
    assert outcome.status == "uncertified_optimal_inaccurate"
    assert np.isnan(outcome.value)
