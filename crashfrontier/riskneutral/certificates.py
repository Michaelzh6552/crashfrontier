from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from crashfrontier.config import LPSettings
from crashfrontier.riskneutral.lp import (
    ConstraintSystem,
    MeasureProblem,
    SupportResult,
    solve_lp,
)

__all__ = ["DualCertificate", "dual_certificate", "interior_margin"]

_DEFAULT_WEIGHT_CUTOFF = 1e-8
_ETA_UPPER = 0.5
_QUOTE_PREFIX = "quote_"
_SCALE_FUN_TOLERANCE = 1e-15
_TIGHT_RESIDUAL = 1e-8
_QUOTE_POSITION_COLUMNS = (
    "quote_id",
    "cp_flag",
    "strike",
    "side",
    "weight",
    "price",
    "norm_price",
)
_STRUCTURAL_COLUMNS = ("constraint", "weight")
_FRAME_COLUMNS = ("kind", "label", "weight")


@dataclass(frozen=True)
class DualCertificate:
    """Static super-replicating portfolio for one support direction.

    Nonzero rows in ``quote_positions`` are the binding quotes that pin the
    frontier in this direction. ``cash + forward·X + Σ weight·payoff``
    super-replicates ``u_p 1{X≤K} + u_L (K-X)^+``; the portfolio cost equals
    ``support_value``. Bid weights are signed (negative = sell at bid) so that
    ``cash + forward + Σ weight·norm_price`` matches that cost on hard systems.
    """

    direction: np.ndarray
    support_value: float
    upper_value: float
    duality_gap: float
    point: np.ndarray
    primal_dual_gap: float
    max_stationarity_residual: float
    max_complementarity_residual: float
    cash: float
    forward: float
    quote_positions: pd.DataFrame
    structural: pd.DataFrame
    threshold_atom: float

    def to_frame(self) -> pd.DataFrame:
        """Stack cash, forward, quote, structural, and atom rows as ``kind, label, weight``."""
        records: list[dict[str, object]] = [
            {"kind": "cash", "label": "cash", "weight": float(self.cash)},
            {"kind": "forward", "label": "forward", "weight": float(self.forward)},
        ]
        for row in self.quote_positions.itertuples(index=False):
            records.append(
                {
                    "kind": "quote",
                    "label": f"{row.quote_id} {row.side}",
                    "weight": float(row.weight),
                }
            )
        for row in self.structural.itertuples(index=False):
            records.append(
                {
                    "kind": "structural",
                    "label": str(row.constraint),
                    "weight": float(row.weight),
                }
            )
        records.append(
            {
                "kind": "threshold_atom",
                "label": "threshold_atom",
                "weight": float(self.threshold_atom),
            }
        )
        return pd.DataFrame.from_records(records, columns=list(_FRAME_COLUMNS))


def interior_margin(problem: MeasureProblem, settings: LPSettings) -> tuple[bool, float, str]:
    """Maximize the uniform quote-interior margin η on the hard (zero-slack) system.

    Each ask row is tightened to ``c_i + width_i η ≤ norm_u_i`` and each bid
    row to ``-c_i + width_i η ≤ -norm_l_i``, with ``width_i = norm_u_i - norm_l_i``
    and ``η ∈ [0, 1/2]``. The interior condition holds when every ``width_i > 0``
    and ``η* > LPSettings.interior_cutoff``. A zero-width quote forces ``η* = 0``.
    """
    n_vars = int(problem.p_target.size)
    widths = _quote_widths(problem)
    eta_column = np.zeros(problem.A_ub.shape[0], dtype=float)
    quote_start = int(problem.quote_row_start)
    for index, width in enumerate(widths):
        eta_column[quote_start + 2 * index] = float(width)
        eta_column[quote_start + 2 * index + 1] = float(width)
    upper_bounds = sparse.hstack(
        [problem.A_ub, sparse.csr_matrix(eta_column.reshape(-1, 1))],
        format="csr",
    )
    equalities = sparse.hstack(
        [problem.A_eq, sparse.csr_matrix((problem.A_eq.shape[0], 1))],
        format="csr",
    )
    objective = np.zeros(n_vars + 1, dtype=float)
    objective[-1] = -1.0
    bounds = list(problem.bounds)
    for index in range(problem.slack_lower.start, problem.slack_upper.stop):
        bounds[index] = (0.0, 0.0)
    bounds.append((0.0, _ETA_UPPER))
    result = solve_lp(
        objective,
        upper_bounds,
        problem.b_ub,
        equalities,
        problem.b_eq,
        tuple(bounds),
        settings,
    )
    message = str(result.message)
    if (
        not result.success
        or result.x is None
        or result.x.size != n_vars + 1
        or not np.isfinite(result.x[-1])
    ):
        return False, float("nan"), message
    eta_star = float(np.clip(result.x[-1], 0.0, _ETA_UPPER))
    if np.any(widths <= 0.0):
        eta_star = 0.0
    return True, eta_star, message


def dual_certificate(
    system: ConstraintSystem,
    result: SupportResult,
    weight_cutoff: float = _DEFAULT_WEIGHT_CUTOFF,
) -> DualCertificate:
    """Read a static super-replicating certificate from a primary support solve.

    SciPy ``ineqlin.marginals`` / ``eqlin.marginals`` are ``∂fun/∂b`` for the
    *minimized* objective ``-target`` (inequality duals typically ≤ 0). They
    are negated and multiplied by ``scale = max(max|u|, 1)`` so that cash,
    forward, and quote weights refer to the unscaled *maximization* problem.
    Ask weight is ``λ``; bid weight is ``-λ`` (sell at bid). Rows with
    ``|λ| > weight_cutoff`` or inequality residual ``< 1e-8`` are listed.
    ``primal_dual_gap`` is the min-problem gap ``|fun - (b_ub·λ + b_eq·ν)|``;
    ``duality_gap`` is the scaled max-problem gap from ``SupportResult``.
    Stationarity uses ``c - A_ubᵀλ - A_eqᵀν - bound duals``; complementarity
    is ``max|λ ⊙ residual|``.
    """
    if not result.success or result.primary.x is None:
        raise ValueError(f"support result is not usable: {result.message}")
    problem = system.problem
    primary = result.primary
    n_rows = int(system.A_ub.shape[0])
    n_vars = int(problem.p_target.size)
    if len(system.row_names) != n_rows:
        raise ValueError("constraint row_names length does not match A_ub")
    lambda_scipy = _as_1d(primary.ineqlin.marginals, n_rows, "ineqlin.marginals")
    nu_scipy = _as_1d(primary.eqlin.marginals, int(problem.A_eq.shape[0]), "eqlin.marginals")
    if nu_scipy.size != 2:
        raise ValueError("expected two equality multipliers (mass, first moment)")
    lower = _as_1d(primary.lower.marginals, n_vars, "lower.marginals")
    upper = _as_1d(primary.upper.marginals, n_vars, "upper.marginals")
    residual = _as_1d(primary.ineqlin.residual, n_rows, "ineqlin.residual")
    primal = _as_1d(primary.x, n_vars, "primary.x")

    dual_stationarity = np.asarray(
        system.A_ub.T @ lambda_scipy + problem.A_eq.T @ nu_scipy + lower + upper,
        dtype=float,
    ).reshape(-1)
    direction_over_scale = _direction_over_scale(problem, dual_stationarity)
    scale = _support_scale(result, direction_over_scale)
    objective = -(
        direction_over_scale[0] * problem.p_target + direction_over_scale[1] * problem.loss_target
    )
    stationarity = np.asarray(objective - dual_stationarity, dtype=float).reshape(-1)
    complementarity = lambda_scipy * residual
    displayed = -scale * lambda_scipy
    cash = float(-scale * nu_scipy[0])
    forward = float(-scale * nu_scipy[1])
    dual_min = _stable_dot(system.b_ub, lambda_scipy) + _stable_dot(problem.b_eq, nu_scipy)
    quotes = _quote_lookup(problem.quotes)
    quote_records: list[dict[str, object]] = []
    structural_records: list[dict[str, object]] = []
    cutoff = float(weight_cutoff)
    for index, name in enumerate(system.row_names):
        parsed = _parse_quote_row(name)
        multiplier = float(displayed[index])
        if parsed is not None:
            quote_id, side = parsed
            if not _keep_row(multiplier, residual[index], cutoff):
                continue
            weight = multiplier if side == "ask" else -multiplier
            quote_records.append(_quote_record(quotes, quote_id, side, weight))
            continue
        if _keep_row(multiplier, residual[index], cutoff):
            structural_records.append({"constraint": name, "weight": multiplier})

    return DualCertificate(
        direction=direction_over_scale * scale,
        support_value=float(result.value),
        upper_value=float(result.upper_value),
        duality_gap=float(result.duality_gap),
        point=np.asarray(result.point, dtype=float).copy(),
        primal_dual_gap=abs(float(primary.fun) - dual_min),
        max_stationarity_residual=float(np.max(np.abs(stationarity))) if stationarity.size else 0.0,
        max_complementarity_residual=(
            float(np.max(np.abs(complementarity))) if complementarity.size else 0.0
        ),
        cash=cash,
        forward=forward,
        quote_positions=_frame(quote_records, _QUOTE_POSITION_COLUMNS),
        structural=_frame(structural_records, _STRUCTURAL_COLUMNS),
        threshold_atom=float(primal[problem.atom_index]),
    )


def _quote_widths(problem: MeasureProblem) -> np.ndarray:
    return (
        problem.quotes["norm_u"].to_numpy(dtype=float)
        - problem.quotes["norm_l"].to_numpy(dtype=float)
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


def _as_1d(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} has length {array.size}, expected {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite entry")
    return array


def _support_scale(result: SupportResult, direction_over_scale: np.ndarray) -> float:
    objective_value = float(result.primary.fun)
    if abs(objective_value) > _SCALE_FUN_TOLERANCE:
        scale = -float(result.value) / objective_value
        if np.isfinite(scale) and scale > 0.0:
            return float(scale)
    return max(float(np.max(np.abs(direction_over_scale))), 1.0)


def _direction_over_scale(problem: MeasureProblem, dual_stationarity: np.ndarray) -> np.ndarray:
    matrix = np.column_stack([problem.p_target, problem.loss_target])
    coefficients, *_ = np.linalg.lstsq(matrix, -dual_stationarity, rcond=None)
    return np.asarray(coefficients, dtype=float).reshape(2)


def _quote_lookup(quotes: pd.DataFrame) -> dict[str, pd.Series]:
    table: dict[str, pd.Series] = {}
    for _, row in quotes.iterrows():
        key = str(row["quote_id"])
        if key not in table:
            table[key] = row
    return table


def _parse_quote_row(name: str) -> tuple[str, str] | None:
    if not name.startswith(_QUOTE_PREFIX):
        return None
    body = name[len(_QUOTE_PREFIX) :]
    if body.endswith("_ask"):
        return body[: -len("_ask")], "ask"
    if body.endswith("_bid"):
        return body[: -len("_bid")], "bid"
    return None


def _keep_row(multiplier: float, residual: float, cutoff: float) -> bool:
    return abs(multiplier) > cutoff or float(residual) < _TIGHT_RESIDUAL


def _quote_record(
    quotes: dict[str, pd.Series],
    quote_id: str,
    side: str,
    weight: float,
) -> dict[str, object]:
    if quote_id not in quotes:
        return {
            "quote_id": quote_id,
            "cp_flag": "",
            "strike": float("nan"),
            "side": side,
            "weight": weight,
            "price": float("nan"),
            "norm_price": float("nan"),
        }
    row = quotes[quote_id]
    ask = side == "ask"
    return {
        "quote_id": quote_id,
        "cp_flag": str(row["cp_flag"]),
        "strike": float(row["strike"]),
        "side": side,
        "weight": weight,
        "price": float(row["ask"] if ask else row["bid"]),
        "norm_price": float(row["norm_u"] if ask else row["norm_l"]),
    }


def _frame(records: list[dict[str, object]], columns: tuple[str, ...]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame.from_records(records, columns=list(columns))
