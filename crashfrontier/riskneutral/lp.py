from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

from crashfrontier.config import LPSettings

__all__ = [
    "ConstraintSystem",
    "MeasureProblem",
    "SupportResult",
    "build_problem",
    "configure_constraints",
    "impose_support_cap",
    "marginal_bounds",
    "minimum_slack",
    "portfolio_bounds",
    "scenario_feasible",
    "solve_lp",
    "support",
]

_SOLUTION_NORM_LIMIT = 1e6
_VALIDATION_ABS_TOLERANCE = 5e-10

_BOUNDARY_DECIMALS = 12
_CELL_STRIKE_TOLERANCE = 1e-11
_REQUIRED_QUOTE_COLUMNS = (
    "x",
    "norm_l",
    "norm_u",
    "strike",
    "quote_id",
    "cp_flag",
    "bid",
    "ask",
)
_SLACK_BUDGET_PADDING = 1e-10
_CAP_STRIKE_ATOL = 5e-12
_POSITIVE_BID_TOLERANCE = 1e-10
_UNKNOWN_MODEL_STATUS = "model_status is Unknown"


@dataclass(frozen=True)
class MeasureProblem:
    """Cell-measure LP with quote slacks and a threshold atom."""

    quotes: pd.DataFrame
    K: float
    boundaries: np.ndarray
    threshold_cell: int
    n_cells: int
    n_quotes: int
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    bounds: tuple[tuple[float, float | None], ...]
    row_names: tuple[str, ...]
    quote_row_start: int
    slack_lower: slice
    slack_upper: slice
    atom_index: int
    p_target: np.ndarray
    loss_target: np.ndarray
    slack_objective: np.ndarray


@dataclass(frozen=True)
class ConstraintSystem:
    """Hard or relaxed constraint set built from a ``MeasureProblem``."""

    problem: MeasureProblem
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    row_names: tuple[str, ...]
    mode: str
    minimum_slack: float
    budget: float | None


@dataclass(frozen=True)
class SupportResult:
    """Primal support value, dual envelope, and chosen (p, L) point."""

    success: bool
    point: np.ndarray
    primary_point: np.ndarray
    value: float
    upper_value: float
    duality_gap: float
    x: np.ndarray | None
    primary: OptimizeResult
    secondary: OptimizeResult | None
    face_shortfall: float
    message: str


def _threshold_cell(boundaries: np.ndarray, threshold: float) -> int:
    """Index of the unique boundary nearest to ``threshold`` after 12-decimal rounding."""
    rounded = float(np.round(float(threshold), decimals=_BOUNDARY_DECIMALS))
    distances = np.abs(np.asarray(boundaries, dtype=float) - rounded)
    order = np.argsort(distances, kind="mergesort")
    best = int(order[0])
    nearest = float(distances[best])
    tie_gap = 0.5 * 10.0 ** (-_BOUNDARY_DECIMALS)
    if len(order) > 1 and float(distances[int(order[1])]) <= nearest + tie_gap:
        raise ValueError("threshold boundary is not unique")
    if nearest > tie_gap:
        raise ValueError("threshold boundary is not unique")
    return best


def build_problem(
    quotes: pd.DataFrame,
    K: float,
    full_strike_count: int | None = None,
    support_cap: float | None = None,
) -> MeasureProblem:
    """Assemble the finite-support LP for risk-neutral crash measures.

    Parameters
    ----------
    quotes
        Quote table with columns ``x, norm_l, norm_u, strike, quote_id,
        cp_flag, bid, ask``.
    K
        Crash threshold in normalized spot units.
    full_strike_count
        Distinct-strike count of the enclosing quote set, used in slack
        weights. Defaults to the number of distinct strikes in ``quotes``.
    support_cap
        When set, drop quotes at or above the cap and insert the cap as a
        cell boundary. The caller still adds ``m_J <= cap z_J``.
    """
    missing = [column for column in _REQUIRED_QUOTE_COLUMNS if column not in quotes.columns]
    if missing:
        raise ValueError(f"quote columns missing: {missing}")
    threshold = float(K)
    if not np.isfinite(threshold):
        raise ValueError("K must be finite")
    panel = quotes.copy()
    finite = (
        np.isfinite(panel["x"])
        & np.isfinite(panel["norm_l"])
        & np.isfinite(panel["norm_u"])
        & (panel["norm_l"] <= panel["norm_u"])
    )
    panel = panel.loc[finite].reset_index(drop=True)
    if int(panel["x"].nunique()) < 2:
        raise ValueError("at least two distinct quote strikes are required")

    cap = None if support_cap is None else float(support_cap)
    if cap is not None:
        if not np.isfinite(cap) or cap <= max(1.0, threshold):
            raise ValueError("support_cap must be finite and exceed max(1, K)")
        strikes = panel["x"].to_numpy(dtype=float)
        outside = strikes >= cap - _CAP_STRIKE_ATOL
        if bool(np.any(outside & (panel["norm_l"].to_numpy(dtype=float) > _POSITIVE_BID_TOLERANCE))):
            raise ValueError("positive call bid is incompatible with support cap")
        panel = panel.loc[~outside].reset_index(drop=True)
        if int(panel["x"].nunique()) < 2:
            raise ValueError("support cap leaves fewer than two strikes")

    x_values = panel["x"].to_numpy(dtype=float)
    rounded_threshold = float(np.round(threshold, decimals=_BOUNDARY_DECIMALS))
    knots = [0.0, rounded_threshold]
    if cap is not None:
        knots.append(float(cap))
    boundaries = np.unique(
        np.round(np.r_[0.0, x_values, np.asarray(knots, dtype=float)], decimals=_BOUNDARY_DECIMALS)
    )
    threshold_cell = _threshold_cell(boundaries, threshold)
    n_cells = int(boundaries.size)
    n_quotes = int(len(panel))
    last_cell = n_cells - 1
    mass_index = n_cells
    slack_lower_start = 2 * n_cells
    slack_upper_start = slack_lower_start + n_quotes
    atom_index = slack_upper_start + n_quotes
    n_vars = atom_index + 1
    n_atom_rows = 1 + int(threshold_cell < last_cell)
    n_rows = (2 * n_cells - 1) + 2 * n_quotes + n_atom_rows

    matrix = np.zeros((n_rows, n_vars), dtype=float)
    rhs = np.zeros(n_rows, dtype=float)
    names: list[str] = []
    row = 0
    for cell, left in enumerate(boundaries):
        matrix[row, cell] = float(left)
        matrix[row, mass_index + cell] = -1.0
        names.append(f"cell_{cell}_moment_lower")
        row += 1
        if cell < last_cell:
            matrix[row, mass_index + cell] = 1.0
            matrix[row, cell] = -float(boundaries[cell + 1])
            names.append(f"cell_{cell}_moment_upper")
            row += 1

    quote_row_start = row
    norm_l = panel["norm_l"].to_numpy(dtype=float)
    norm_u = panel["norm_u"].to_numpy(dtype=float)
    quote_ids = panel["quote_id"].astype(str).to_numpy()
    for index, strike in enumerate(x_values):
        active = boundaries >= float(strike) - _CELL_STRIKE_TOLERANCE
        call = np.zeros(n_vars, dtype=float)
        call[mass_index : mass_index + n_cells] = active.astype(float)
        call[:n_cells] = np.where(active, -float(strike), 0.0)
        matrix[row] = call
        matrix[row, slack_upper_start + index] = -1.0
        rhs[row] = float(norm_u[index])
        names.append(f"quote_{quote_ids[index]}_ask")
        row += 1
        matrix[row] = -call
        matrix[row, slack_lower_start + index] = -1.0
        rhs[row] = -float(norm_l[index])
        names.append(f"quote_{quote_ids[index]}_bid")
        row += 1

    matrix[row, atom_index] = 1.0
    matrix[row, threshold_cell] = -1.0
    names.append("threshold_atom_mass")
    row += 1
    if threshold_cell < last_cell:
        right = float(boundaries[threshold_cell + 1])
        matrix[row, mass_index + threshold_cell] = 1.0
        matrix[row, threshold_cell] = -right
        matrix[row, atom_index] = right - threshold
        names.append("threshold_atom_residual_upper")
        row += 1
    if row != n_rows:
        raise RuntimeError("inequality row count does not match allocation")

    equality = np.zeros((2, n_vars), dtype=float)
    equality[0, :n_cells] = 1.0
    equality[1, mass_index:slack_lower_start] = 1.0
    b_eq = np.array([1.0, 1.0], dtype=float)

    p_target = np.zeros(n_vars, dtype=float)
    loss_target = np.zeros(n_vars, dtype=float)
    p_target[:threshold_cell] = 1.0
    p_target[atom_index] = 1.0
    loss_target[:threshold_cell] = threshold
    loss_target[mass_index : mass_index + threshold_cell] = -1.0

    strike_counts = panel.groupby("strike")["strike"].transform("size").to_numpy(dtype=float)
    n_strikes = float(
        panel["strike"].nunique() if full_strike_count is None else full_strike_count
    )
    weights = 1.0 / (n_strikes * strike_counts)
    slack_objective = np.zeros(n_vars, dtype=float)
    slack_objective[slack_lower_start:slack_upper_start] = weights
    slack_objective[slack_upper_start:atom_index] = weights

    return MeasureProblem(
        quotes=panel,
        K=threshold,
        boundaries=boundaries,
        threshold_cell=threshold_cell,
        n_cells=n_cells,
        n_quotes=n_quotes,
        A_ub=sparse.csr_matrix(matrix),
        b_ub=rhs,
        A_eq=sparse.csr_matrix(equality),
        b_eq=b_eq,
        bounds=tuple((0.0, None) for _ in range(n_vars)),
        row_names=tuple(names),
        quote_row_start=quote_row_start,
        slack_lower=slice(slack_lower_start, slack_upper_start),
        slack_upper=slice(slack_upper_start, atom_index),
        atom_index=atom_index,
        p_target=p_target,
        loss_target=loss_target,
        slack_objective=slack_objective,
    )


def solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, settings: LPSettings) -> OptimizeResult:
    """Solve a linear program with HiGHS dual simplex, then interior point."""
    options = {
        "primal_feasibility_tolerance": settings.primal_feasibility_tolerance,
        "dual_feasibility_tolerance": settings.dual_feasibility_tolerance,
        "time_limit": settings.time_limit_seconds,
    }
    arguments = {
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "options": options,
    }
    result = linprog(**arguments, method="highs-ds")
    unknown = _UNKNOWN_MODEL_STATUS in str(result.message)
    if int(result.status) != 4 and not unknown:
        return result
    fallback = linprog(**arguments, method="highs-ipm")
    return fallback if fallback.success else result


def minimum_slack(problem: MeasureProblem, settings: LPSettings) -> OptimizeResult:
    """Minimize the weighted quote-slack objective ``Σ w(ξ⁻ + ξ⁺)``."""
    return solve_lp(
        problem.slack_objective,
        problem.A_ub,
        problem.b_ub,
        problem.A_eq,
        problem.b_eq,
        problem.bounds,
        settings,
    )


def configure_constraints(
    problem: MeasureProblem,
    settings: LPSettings,
    mode: str,
    budget: float | None = None,
) -> ConstraintSystem | None:
    """Build a hard (zero slack) or relaxed (slack-budget) constraint system."""
    minimum = minimum_slack(problem, settings)
    if not minimum.success:
        return None
    delta = float(minimum.fun)
    if mode == "hard":
        bounds = list(problem.bounds)
        for index in range(problem.slack_lower.start, problem.slack_upper.stop):
            bounds[index] = (0.0, 0.0)
        hard = replace(problem, bounds=tuple(bounds))
        feasible = solve_lp(
            np.zeros_like(hard.slack_objective),
            hard.A_ub,
            hard.b_ub,
            hard.A_eq,
            hard.b_eq,
            hard.bounds,
            settings,
        )
        if not feasible.success:
            return None
        return ConstraintSystem(
            problem=hard,
            A_ub=hard.A_ub,
            b_ub=hard.b_ub,
            row_names=hard.row_names,
            mode=mode,
            minimum_slack=delta,
            budget=0.0,
        )
    if mode != "relaxed":
        raise ValueError(f"unrecognized constraint mode {mode!r}")
    if budget is None:
        budget = delta + _SLACK_BUDGET_PADDING
    elif delta > float(budget) + _SLACK_BUDGET_PADDING:
        return None
    slack_row = sparse.csr_matrix(problem.slack_objective.reshape(1, -1))
    return ConstraintSystem(
        problem=problem,
        A_ub=sparse.vstack([problem.A_ub, slack_row], format="csr"),
        b_ub=np.r_[problem.b_ub, float(budget)],
        row_names=tuple(problem.row_names) + ("slack_budget",),
        mode=mode,
        minimum_slack=delta,
        budget=float(budget),
    )


def impose_support_cap(system: ConstraintSystem, cap: float) -> ConstraintSystem:
    """Append ``m_J - cap z_J <= 0`` as row ``support_cap``."""
    problem = system.problem
    tail = problem.n_cells - 1
    n_vars = int(problem.slack_objective.size)
    row = np.zeros((1, n_vars), dtype=float)
    row[0, problem.n_cells + tail] = 1.0
    row[0, tail] = -float(cap)
    return ConstraintSystem(
        problem=problem,
        A_ub=sparse.vstack([system.A_ub, sparse.csr_matrix(row)], format="csr"),
        b_ub=np.r_[system.b_ub, 0.0],
        row_names=tuple(system.row_names) + ("support_cap",),
        mode=system.mode,
        minimum_slack=system.minimum_slack,
        budget=system.budget,
    )


def _stable_dot(left, right) -> float:
    left_values = np.asarray(left, dtype=float).ravel()
    right_values = np.asarray(right, dtype=float).ravel()
    if left_values.shape != right_values.shape:
        raise ValueError("dot-product operands have different shapes")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        return float("nan")
    return math.fsum(
        float(left_value) * float(right_value)
        for left_value, right_value in zip(left_values, right_values)
    )


def _validation_tolerance(settings: LPSettings) -> float:
    return max(
        _VALIDATION_ABS_TOLERANCE,
        float(settings.validation_multiplier) * float(settings.primal_feasibility_tolerance),
    )


def _valid_lp_solution(
    result: OptimizeResult,
    A_ub: sparse.csr_matrix,
    b_ub: np.ndarray,
    A_eq: sparse.csr_matrix,
    b_eq: np.ndarray,
    bounds: tuple[tuple[float, float | None], ...],
    tolerance: float,
) -> bool:
    if (
        not result.success
        or result.x is None
        or not np.isfinite(result.fun)
        or not np.all(np.isfinite(result.x))
    ):
        return False
    x = np.asarray(result.x, dtype=float)
    if float(np.max(np.abs(x))) > _SOLUTION_NORM_LIMIT:
        return False
    equality_error = float(np.max(np.abs(A_eq @ x - b_eq))) if A_eq.shape[0] else 0.0
    inequality_error = float(np.max(A_ub @ x - b_ub)) if A_ub.shape[0] else 0.0
    if equality_error > tolerance or inequality_error > tolerance:
        return False
    for value, (lower, upper) in zip(x, bounds):
        if lower is not None and value < float(lower) - tolerance:
            return False
        if upper is not None and value > float(upper) + tolerance:
            return False
    return True


def _target_point(problem: MeasureProblem, x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            _stable_dot(problem.p_target, x),
            _stable_dot(problem.loss_target, x),
        ],
        dtype=float,
    )


def _failed_support(
    primary: OptimizeResult,
    message: str,
    *,
    primary_point: np.ndarray | None = None,
    secondary: OptimizeResult | None = None,
) -> SupportResult:
    missing = np.full(2, np.nan)
    return SupportResult(
        success=False,
        point=missing.copy(),
        primary_point=missing.copy() if primary_point is None else primary_point,
        value=float("nan"),
        upper_value=float("nan"),
        duality_gap=float("nan"),
        x=None,
        primary=primary,
        secondary=secondary,
        face_shortfall=float("nan"),
        message=message,
    )


def _minimization_dual_value(result: OptimizeResult, b_ub: np.ndarray, b_eq: np.ndarray) -> float:
    """SciPy ``linprog`` dual objective under the ∂fun/∂b sign convention.

    ``ineqlin.marginals`` and ``eqlin.marginals`` are sensitivities of the
    *minimized* objective to the right-hand sides, not textbook λ ≥ 0.
    Stationarity is ``c = A_ubᵀ λ + A_eqᵀ ν + bound duals``, so the min-problem
    dual value is ``b_ub·λ + b_eq·ν`` and matches ``fun`` at complementarity.
    """
    return _stable_dot(b_ub, result.ineqlin.marginals) + _stable_dot(
        b_eq, result.eqlin.marginals
    )


def support(
    system: ConstraintSystem,
    settings: LPSettings,
    direction,
    lexicographic: bool = True,
) -> SupportResult:
    """Maximize ``u · (p, L)`` over measures in ``system``.

    The solver minimizes ``-target`` with ``target = (u_p p_target + u_L
    loss_target) / scale`` and ``scale = max(max|u|, 1)``. Dual values are
    reconstructed from SciPy HiGHS ``marginals`` (∂fun/∂b, typically ≤ 0 on
    inequalities) as ``-(b_ub·λ + b_eq·ν) * scale``. A lexicographic second
    LP, when requested, pins the primary objective at ``value`` and maximizes
    the rotated direction ``(-u_L, u_p) / scale`` to pick an exposed-face
    endpoint.
    """
    problem = system.problem
    original_direction = np.asarray(direction, dtype=float).reshape(-1)
    if original_direction.shape != (2,) or not np.all(np.isfinite(original_direction)):
        raise ValueError(f"support direction must be a finite 2-vector: {original_direction}")
    scale = max(float(np.max(np.abs(original_direction))), 1.0)
    original_target = (
        original_direction[0] * problem.p_target + original_direction[1] * problem.loss_target
    )
    target = original_target / scale
    primary = solve_lp(
        -target,
        system.A_ub,
        system.b_ub,
        problem.A_eq,
        problem.b_eq,
        problem.bounds,
        settings,
    )
    tolerance = _validation_tolerance(settings)
    if not _valid_lp_solution(
        primary,
        system.A_ub,
        system.b_ub,
        problem.A_eq,
        problem.b_eq,
        problem.bounds,
        tolerance,
    ):
        return _failed_support(
            primary,
            f"non-finite or failed primary support solve: {primary.message}",
        )
    value = float(-primary.fun) * scale
    dual = -_minimization_dual_value(primary, system.b_ub, problem.b_eq) * scale
    duality_gap = abs(float(dual) - value)
    upper_value = value + duality_gap
    primary_point = _target_point(problem, primary.x)
    chosen = primary
    secondary: OptimizeResult | None = None
    if lexicographic:
        face = sparse.csr_matrix(original_target.reshape(1, -1))
        A_eq = sparse.vstack([problem.A_eq, face], format="csr")
        b_eq = np.r_[problem.b_eq, value]
        rotated = np.array([-original_direction[1], original_direction[0]], dtype=float) / scale
        orthogonal = rotated[0] * problem.p_target + rotated[1] * problem.loss_target
        secondary = solve_lp(
            -orthogonal,
            system.A_ub,
            system.b_ub,
            A_eq,
            b_eq,
            problem.bounds,
            settings,
        )
        if _valid_lp_solution(
            secondary,
            system.A_ub,
            system.b_ub,
            A_eq,
            b_eq,
            problem.bounds,
            tolerance,
        ):
            chosen = secondary
    point = _target_point(problem, chosen.x)
    if not np.all(np.isfinite(point)):
        return _failed_support(
            primary,
            "support projection is non-finite",
            primary_point=primary_point,
            secondary=secondary,
        )
    face_shortfall = max(0.0, value - _stable_dot(original_direction, point))
    return SupportResult(
        success=True,
        point=point,
        primary_point=primary_point,
        value=value,
        upper_value=upper_value,
        duality_gap=duality_gap,
        x=np.asarray(chosen.x, dtype=float).copy(),
        primary=primary,
        secondary=secondary,
        face_shortfall=face_shortfall,
        message="ok",
    )


def marginal_bounds(
    system: ConstraintSystem, settings: LPSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate bounds from ±e₁, ±e₂ using the duality-gap envelope."""
    lower = []
    upper = []
    for direction in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        high = support(system, settings, direction, lexicographic=False)
        low = support(system, settings, -direction, lexicographic=False)
        if not high.success or not low.success:
            raise RuntimeError("marginal support failed")
        upper.append(float(high.upper_value))
        lower.append(float(-low.upper_value))
    lo = np.maximum(np.asarray(lower, dtype=float), 0.0)
    hi = np.maximum(np.asarray(upper, dtype=float), lo)
    return lo, hi


def portfolio_bounds(
    system: ConstraintSystem,
    settings: LPSettings,
    u_p: float,
    u_L: float,
) -> tuple[float, float]:
    """Normalized price interval ``(-h(-u), h(u))`` for payoff ``u_p, u_L``."""
    direction = np.array([float(u_p), float(u_L)], dtype=float)
    high = support(system, settings, direction, lexicographic=False)
    low = support(system, settings, -direction, lexicographic=False)
    if not high.success or not low.success:
        raise RuntimeError("portfolio support failed")
    return float(-low.value), float(high.value)


def scenario_feasible(
    system: ConstraintSystem,
    settings: LPSettings,
    p: float,
    L: float,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether some feasible measure attains crash targets ``(p, L)``."""
    problem = system.problem
    extra = sparse.csr_matrix(np.vstack([problem.p_target, problem.loss_target]))
    A_eq = sparse.vstack([problem.A_eq, extra], format="csr")
    b_eq = np.r_[problem.b_eq, float(p), float(L)]
    result = solve_lp(
        np.zeros(problem.p_target.size, dtype=float),
        system.A_ub,
        system.b_ub,
        A_eq,
        b_eq,
        problem.bounds,
        settings,
    )
    return _valid_lp_solution(
        result,
        system.A_ub,
        system.b_ub,
        A_eq,
        b_eq,
        problem.bounds,
        float(tolerance),
    )
