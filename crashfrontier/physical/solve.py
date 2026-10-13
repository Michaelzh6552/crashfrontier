from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
from cvxpy.constraints.second_order import SOC

warnings.filterwarnings(
    "ignore",
    message=r"Solution may be inaccurate.*",
    category=UserWarning,
    module=r"cvxpy\.problems\.problem",
)

__all__ = [
    "ConicOutcome",
    "ConicSettings",
    "max_constraint_violation",
    "solve_with_rescue",
    "solver_options",
]

_SCS_EPS = 1e-6
_SCS_MAX_ITERS = 20_000
_CERTIFIED_INACCURATE = "optimal_inaccurate_certified"
_UNCERTIFIED_INACCURATE = "uncertified_optimal_inaccurate"
_SOLVER_ERROR = "solver_error"
_OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_TERMINAL_STATUSES = {
    cp.INFEASIBLE,
    cp.UNBOUNDED,
    cp.UNBOUNDED_INACCURATE,
}
_UNCERTIFIED_INFEASIBLE = "uncertified_infeasible_inaccurate"
_BRACKET_BOUND = "depth_bracket_bound"
_DEPTH_CUT_TOLERANCE = 1e-8
_DEPTH_BRACKET_TOLERANCE = 1e-6
_DEPTH_BISECTION_LIMIT = 40
_INFEASIBLE_STATUSES = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
_FAILURE_STATUSES = {
    _SOLVER_ERROR,
    _UNCERTIFIED_INACCURATE,
    _UNCERTIFIED_INFEASIBLE,
}


@dataclass(frozen=True)
class ConicSettings:
    """SOCP solver names, tolerances, and inaccurate-status rescue rules."""

    solver: str = "CLARABEL"
    fallbacks: tuple[str, ...] = ("ECOS", "SCS")
    tolerance: float = 1e-9
    max_iterations: int = 2000
    violation_tolerance: float = 1e-6
    objective_tolerance: float = 1e-7
    warm_start: bool = True


@dataclass(frozen=True)
class ConicOutcome:
    """Primal value, solver identity, and constraint residual of one SOCP."""

    ok: bool
    value: float
    status: str
    solver: str
    violation: float
    iterations: int | None
    solve_time: float | None


def solver_options(name: str, settings: ConicSettings | None = None) -> dict[str, float | int]:
    """Keyword arguments for ``problem.solve`` under one solver name."""
    options = ConicSettings() if settings is None else settings
    key = str(name).upper()
    if key == "CLARABEL":
        return {
            "tol_gap_abs": options.tolerance,
            "tol_gap_rel": options.tolerance,
            "tol_feas": options.tolerance,
            "max_iter": options.max_iterations,
        }
    if key == "ECOS":
        return {
            "abstol": options.tolerance,
            "reltol": options.tolerance,
            "feastol": options.tolerance,
            "max_iters": options.max_iterations,
        }
    if key == "SCS":
        return {"eps": _SCS_EPS, "max_iters": _SCS_MAX_ITERS}
    return {}


def max_constraint_violation(constraints) -> float:
    """Largest primal residual; SOC cones use ``max(||t|| - s, 0)``."""
    worst = 0.0
    for constraint in constraints:
        residual = _constraint_residual(constraint)
        if not np.isfinite(residual):
            return float("inf")
        if residual > worst:
            worst = residual
    return float(worst)


def solve_with_rescue(
    problem: cp.Problem,
    constraints,
    settings: ConicSettings,
    *,
    warm_start: bool,
) -> ConicOutcome:
    """Solve an SOCP, rescuing inaccurate optimal and infeasible statuses.

    ``OPTIMAL_INACCURATE`` and ``INFEASIBLE_INACCURATE`` both cold-start the
    primary solver and then try installed fallbacks. Two inaccurate optimal
    values are accepted only when they agree within
    ``objective_tolerance * (1+|v|)``. Infeasibility is accepted only for a
    definitive ``INFEASIBLE`` status, or when a second attempt also returns
    ``INFEASIBLE_INACCURATE``. A later optimal result replaces an inaccurate
    infeasibility. ``ok`` also requires a finite value and a violation within
    tolerance. Variable values on ``problem`` are restored to the chosen attempt.
    """
    first, first_values = _attempt(
        problem, constraints, settings, settings.solver, warm_start
    )
    if first.ok and first.status != cp.OPTIMAL_INACCURATE:
        _restore_values(first_values)
        return first
    if first.status in _TERMINAL_STATUSES:
        _restore_values(first_values)
        return first

    records = [(first, first_values)]
    if first.status in {cp.OPTIMAL_INACCURATE, cp.INFEASIBLE_INACCURATE}:
        cold, cold_values = _attempt(problem, constraints, settings, settings.solver, False)
        records.append((cold, cold_values))
        if cold.ok and cold.status == cp.OPTIMAL:
            _restore_values(cold_values)
            return cold
        if first.status == cp.OPTIMAL_INACCURATE:
            certified = _certified_pair(first, cold, settings)
            if certified is not None:
                _restore_certified(certified, first, first_values, cold, cold_values)
                return certified

    installed = {str(name).upper() for name in cp.installed_solvers()}
    primary = str(settings.solver).upper()
    for fallback in settings.fallbacks:
        name = str(fallback).upper()
        if name == primary or name not in installed:
            continue
        cross, cross_values = _attempt(problem, constraints, settings, name, False)
        records.append((cross, cross_values))
        if cross.ok and cross.status == cp.OPTIMAL:
            _restore_values(cross_values)
            return cross

    usable = [
        (candidate, values)
        for candidate, values in records
        if candidate.ok
        and candidate.status == cp.OPTIMAL_INACCURATE
        and np.isfinite(candidate.value)
    ]
    for index, (left, left_values) in enumerate(usable):
        for right, right_values in usable[index + 1 :]:
            certified = _certified_pair(left, right, settings)
            if certified is not None:
                _restore_certified(certified, left, left_values, right, right_values)
                return certified
    confirmed = _confirmed_infeasible(records)
    if confirmed is not None:
        return confirmed
    _restore_values(first_values)
    if any(record.status == cp.OPTIMAL_INACCURATE for record, _ in records):
        status = _UNCERTIFIED_INACCURATE
    elif any(record.status == cp.INFEASIBLE_INACCURATE for record, _ in records):
        status = _UNCERTIFIED_INFEASIBLE
    else:
        status = str(first.status)
    return replace(first, ok=False, value=float("nan"), status=status)


def finite_expression(expression) -> float | None:
    """Finite scalar value of a solved CVXPY expression, if one is available."""
    raw = getattr(expression, "value", None)
    if raw is None:
        return None
    try:
        array = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.isfinite(array[0]):
        return None
    return float(array[0])


def depth_cut_decision(
    outcome: ConicOutcome,
    achieved: float | None,
    sense: str,
    mid: float,
    *,
    tolerance: float,
) -> str:
    """Classify a depth-cut probe as ``attained``, ``rejected``, or ``failure``.

    The feasible side moves only when the solved depth meets the cut within
    ``tolerance``. A loose constraint residual does not count as extra depth.
    """
    status = str(outcome.status)
    if status in _FAILURE_STATUSES:
        return "failure"
    if status in _INFEASIBLE_STATUSES or status == _UNCERTIFIED_INACCURATE:
        return "rejected"
    meets = _depth_cut_met(achieved, sense, mid, tolerance)
    if meets is False:
        return "rejected"
    if outcome.ok and (meets is True or float(outcome.violation) <= tolerance):
        return "attained"
    if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, _CERTIFIED_INACCURATE}:
        return "rejected"
    return "failure"


def bisect_feasible_bound(
    *,
    sense: str,
    upper: float,
    probe,
    certify_endpoint,
    solver: str,
    tolerance: float = _DEPTH_CUT_TOLERANCE,
    bracket_tolerance: float = _DEPTH_BRACKET_TOLERANCE,
    step_limit: int = _DEPTH_BISECTION_LIMIT,
) -> ConicOutcome:
    """Bisect a monotone feasibility cut on ``[0, upper]``.

    The bracket begins at the endpoint already enforced by the depth
    constraints: ``0`` for a maximum and ``upper`` for a minimum. A solver
    failure returns the conservative outer side of the current bracket. When
    every stricter cut is rejected and the base system is feasible, the
    logical endpoint is returned.
    Cuts are accepted only within ``tolerance`` of the solved depth. The search
    stops once the bracket is inside ``bracket_tolerance``, so the final
    endpoint is not asked to separate a numerically singular inequality.
    """
    if sense not in {"min", "max"}:
        raise ValueError(f"unknown depth sense {sense!r}")
    # Certify the uncut system before a sequence of deliberately infeasible
    # probes can leave shared CVXPY variables without a usable primal value.
    # The cached outcome is sufficient for the exact endpoint fallback.
    base = certify_endpoint()
    if not base.ok:
        if str(base.status) in _INFEASIBLE_STATUSES:
            return _bisect_failed(solver)
        return replace(base, ok=False, value=float("nan"))
    low = 0.0
    high = float(upper)
    best: ConicOutcome | None = None
    for _ in range(int(step_limit)):
        if high - low <= bracket_tolerance:
            break
        mid = 0.5 * (low + high)
        outcome, achieved = probe(mid)
        decision = depth_cut_decision(outcome, achieved, sense, mid, tolerance=tolerance)
        if decision == "failure":
            boundary = high if sense == "max" else low
            return replace(
                base,
                ok=True,
                value=float(boundary),
                status=_BRACKET_BOUND,
            )
        if decision == "attained":
            if sense == "max":
                low = mid
            else:
                high = mid
            best = replace(outcome, value=float(mid))
            continue
        if sense == "max":
            high = mid
        else:
            low = mid
    if best is not None:
        boundary = high if sense == "max" else low
        return _snap_depth_outcome(
            replace(best, value=float(boundary), status=_BRACKET_BOUND),
            upper,
            tolerance,
            solver,
        )
    collapsed = (
        high <= bracket_tolerance
        if sense == "max"
        else low >= float(upper) - bracket_tolerance
    )
    if not collapsed:
        return _bisect_failed(solver)
    endpoint = 0.0 if sense == "max" else float(upper)
    return replace(base, ok=True, value=endpoint)


def _confirmed_infeasible(records) -> ConicOutcome | None:
    exact = [
        (outcome, values) for outcome, values in records if outcome.status == cp.INFEASIBLE
    ]
    if exact:
        outcome, values = exact[0]
        _restore_values(values)
        return outcome
    inaccurate = [
        (outcome, values)
        for outcome, values in records
        if outcome.status == cp.INFEASIBLE_INACCURATE
    ]
    if len(inaccurate) < 2:
        return None
    for index, (left, left_values) in enumerate(inaccurate):
        for right, _right_values in inaccurate[index + 1 :]:
            if str(left.solver).upper() == str(right.solver).upper():
                continue
            _restore_values(left_values)
            return replace(
                left,
                ok=False,
                status=cp.INFEASIBLE_INACCURATE,
                solver=f"{left.solver}|{right.solver}",
            )
    return None


def _depth_cut_met(
    achieved: float | None, sense: str, mid: float, tolerance: float
) -> bool | None:
    if achieved is None or not np.isfinite(achieved):
        return None
    if sense == "max":
        return float(achieved) + tolerance >= float(mid)
    if sense == "min":
        return float(achieved) <= float(mid) + tolerance
    raise ValueError(f"unknown depth sense {sense!r}")


def _snap_depth_outcome(
    best: ConicOutcome, upper: float, tolerance: float, solver: str
) -> ConicOutcome:
    value = float(best.value)
    if not np.isfinite(value) or value < -tolerance or value > float(upper) + tolerance:
        return _bisect_failed(solver)
    value = min(max(value, 0.0), float(upper))
    return replace(best, ok=True, value=value)


def _bisect_failed(solver: str) -> ConicOutcome:
    return ConicOutcome(
        ok=False,
        value=float("nan"),
        status="bisect_failed",
        solver=solver,
        violation=float("inf"),
        iterations=None,
        solve_time=None,
    )


def _attempt(
    problem: cp.Problem,
    constraints,
    settings: ConicSettings,
    solver: str,
    warm_start: bool,
) -> tuple[ConicOutcome, list[tuple[object, np.ndarray | None]]]:
    name = str(solver).upper()
    try:
        problem.solve(
            solver=name,
            warm_start=warm_start,
            verbose=False,
            **solver_options(name, settings),
        )
    except cp.error.SolverError:
        return (
            ConicOutcome(
                ok=False,
                value=float("nan"),
                status=_SOLVER_ERROR,
                solver=name,
                violation=float("inf"),
                iterations=None,
                solve_time=None,
            ),
            [],
        )
    value = _finite_value(problem.value)
    violation = max_constraint_violation(constraints)
    status = str(problem.status)
    ok = bool(
        status in _OPTIMAL_STATUSES
        and np.isfinite(value)
        and violation <= settings.violation_tolerance
    )
    iterations, solve_time = _solver_stats(problem)
    outcome = ConicOutcome(
        ok=ok,
        value=value,
        status=status,
        solver=name,
        violation=violation,
        iterations=iterations,
        solve_time=solve_time,
    )
    return outcome, _capture_values(problem)


def _certified_pair(
    first: ConicOutcome, second: ConicOutcome, settings: ConicSettings
) -> ConicOutcome | None:
    if not (
        first.ok
        and second.ok
        and first.status == cp.OPTIMAL_INACCURATE
        and second.status == cp.OPTIMAL_INACCURATE
        and np.isfinite(first.value)
        and np.isfinite(second.value)
    ):
        return None
    gap = settings.objective_tolerance * (1.0 + abs(first.value))
    if abs(first.value - second.value) > gap:
        return None
    chosen = first if first.violation <= second.violation else second
    return replace(
        chosen,
        status=_CERTIFIED_INACCURATE,
        solver=f"{first.solver}|{second.solver}",
    )


def _capture_values(problem) -> list[tuple[object, np.ndarray | None]]:
    variables = getattr(problem, "variables", None)
    if not callable(variables):
        return []
    captured: list[tuple[object, np.ndarray | None]] = []
    for variable in variables():
        value = getattr(variable, "value", None)
        captured.append(
            (variable, None if value is None else np.array(value, copy=True))
        )
    return captured


def _restore_values(snapshot: list[tuple[object, np.ndarray | None]]) -> None:
    for variable, value in snapshot:
        if hasattr(variable, "value"):
            variable.value = value


def _restore_certified(
    _certified: ConicOutcome,
    left: ConicOutcome,
    left_values,
    right: ConicOutcome,
    right_values,
) -> None:
    if left.violation <= right.violation:
        _restore_values(left_values)
        return
    _restore_values(right_values)


def _constraint_residual(constraint) -> float:
    if isinstance(constraint, SOC):
        return _soc_residual(constraint)
    try:
        raw = constraint.violation()
    except ValueError:
        return float("inf")
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.max(values))


def _soc_residual(constraint: SOC) -> float:
    scalar = constraint.args[0].value
    vector = constraint.args[1].value
    if scalar is None or vector is None:
        return float("inf")
    cone_scalar = np.asarray(scalar, dtype=float).reshape(-1)
    cone_vector = np.asarray(vector, dtype=float).reshape(-1)
    if (
        cone_scalar.size == 0
        or cone_vector.size == 0
        or not np.all(np.isfinite(cone_scalar))
        or not np.all(np.isfinite(cone_vector))
    ):
        return float("inf")
    return max(float(np.linalg.norm(cone_vector) - float(cone_scalar[0])), 0.0)


def _finite_value(value) -> float:
    if value is None:
        return float("nan")
    numeric = float(value)
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def _solver_stats(problem: cp.Problem) -> tuple[int | None, float | None]:
    stats = problem.solver_stats
    if stats is None:
        return None, None
    iterations = getattr(stats, "num_iters", None)
    solve_time = getattr(stats, "solve_time", None)
    return (
        int(iterations) if iterations is not None else None,
        float(solve_time) if solve_time is not None and np.isfinite(solve_time) else None,
    )
