from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd

from crashfrontier.errors import SolverFailure
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.physical.gamma2 import ConicPoint
from crashfrontier.physical.solve import (
    ConicOutcome,
    ConicSettings,
    bisect_feasible_bound,
    finite_expression,
    solve_with_rescue,
)

__all__ = ["Gamma1Model"]

_REQUIRED_QUOTE_COLUMNS = ("x", "norm_l", "norm_u")
_BOUNDARY_DECIMALS = 12
_CAP_STRIKE_ATOL = 5e-12
_POSITIVE_BID_TOLERANCE = 1e-10
_THRESHOLD_MATCH_ATOL = 5e-12
_CELL_STRIKE_TOLERANCE = 1e-11
_BOUND_LOWER_TOLERANCE = 1e-6
_BOUND_UPPER_TOLERANCE = 1e-6
_BOUND_ORDER_TOLERANCE = 1e-8
_INTERIOR_MARGIN_CUTOFF = 1e-10
_ZERO_WIDTH_TOLERANCE = 1e-12
_ETA_UPPER = 0.5
_DEFAULT_MINIMUM_PROB_UPPER = 1e-4
_DEPTH_OBJECTIVE_SCALE = 100.0
_CERTIFIED_INFEASIBLE = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
_TARGETS = ("prob", "loss")


class Gamma1Model:
    """γ=1 physical crash model with cell mass, mean, and below-threshold second moments."""

    def __init__(
        self,
        quotes: pd.DataFrame,
        K: float,
        *,
        support_cap: float | None = 5.0,
        settings: ConicSettings = ConicSettings(),
    ) -> None:
        threshold = float(K)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("K must lie strictly between zero and one")
        missing = [column for column in _REQUIRED_QUOTE_COLUMNS if column not in quotes.columns]
        if missing:
            raise ValueError(f"quote columns missing: {missing}")
        panel = quotes.copy()
        finite = (
            np.isfinite(panel["x"])
            & np.isfinite(panel["norm_l"])
            & np.isfinite(panel["norm_u"])
            & panel["x"].ge(0.0)
            & (panel["norm_l"] <= panel["norm_u"])
        )
        panel = panel.loc[finite].reset_index(drop=True)
        if int(panel["x"].nunique()) < 2:
            raise ValueError("at least two distinct quote strikes are required")

        cap = None if support_cap is None else float(support_cap)
        if cap is not None:
            if not np.isfinite(cap) or cap <= max(1.0, threshold):
                raise ValueError("support_cap must be finite and exceed max(1, K)")
            outside = panel["x"].ge(cap - _CAP_STRIKE_ATOL)
            if bool((panel.loc[outside, "norm_l"] > _POSITIVE_BID_TOLERANCE).any()):
                raise ValueError("positive call bid is incompatible with support cap")
            panel = panel.loc[~outside].reset_index(drop=True)
            if int(panel["x"].nunique()) < 2:
                raise ValueError("support cap leaves fewer than two strikes")

        boundary_values = np.r_[0.0, panel["x"].to_numpy(dtype=float), threshold]
        if cap is not None:
            boundary_values = np.r_[boundary_values, cap]
        boundaries = np.unique(np.round(boundary_values, decimals=_BOUNDARY_DECIMALS))
        n_cells = int(boundaries.size - 1) if cap is not None else int(boundaries.size)
        cell_starts = boundaries[:n_cells]
        matches = np.flatnonzero(
            np.isclose(cell_starts, threshold, atol=_THRESHOLD_MATCH_ATOL, rtol=0.0)
        )
        if len(matches) != 1:
            raise ValueError("threshold boundary is not unique")

        self.quotes = panel
        self.K = threshold
        self.support_cap = cap
        self.settings = settings
        self.boundaries = boundaries
        self.n_cells = n_cells
        self.cell_starts = cell_starts
        self.threshold_cell = int(matches[0])
        self.z = cp.Variable(n_cells, nonneg=True, name="cell_mass")
        self.m = cp.Variable(n_cells, nonneg=True, name="cell_first_moment")
        self.v = cp.Variable(
            self.threshold_cell,
            nonneg=True,
            name="below_threshold_second_moment",
        )
        self.atom = cp.Variable(nonneg=True, name="threshold_atom")

        linear = [
            cp.sum(self.z) == 1.0,
            cp.sum(self.m) == 1.0,
        ]
        linear.extend(self._linear_cell_constraints(self.z, self.m, self.atom))
        self.linear_structural_constraints = list(linear)
        structural = list(linear)
        structural.extend(self._second_moment_constraints(self.z, self.m, self.v))
        self.structural_constraints = list(structural)

        self.quote_calls: list[cp.Expression] = []
        self.quote_constraints: list[cp.Constraint] = []
        strikes = panel["x"].to_numpy(dtype=float)
        lower = panel["norm_l"].to_numpy(dtype=float)
        upper = panel["norm_u"].to_numpy(dtype=float)
        for strike, bid, ask in zip(strikes, lower, upper):
            payoff = self._call_payoff(self.z, self.m, float(strike))
            self.quote_calls.append(payoff)
            self.quote_constraints.extend(
                [
                    payoff >= float(bid),
                    payoff <= float(ask),
                ]
            )
        constraints = structural + self.quote_constraints
        below_first = cp.sum(self.m[: self.threshold_cell])
        below_second = cp.sum(self.v)
        probability = below_first + self.K * self.atom
        loss = self.K * below_first - below_second
        self.target = cp.hstack([probability, loss])
        constraints.extend(
            [
                probability >= 0.0,
                probability <= 1.0,
                loss >= 0.0,
                loss <= self.K * probability,
            ]
        )
        self.constraints = constraints
        self.direction = cp.Parameter(2, name="gamma1_support_direction")
        self.support_problem = cp.Problem(
            cp.Maximize(self.direction @ self.target),
            self.constraints,
        )
        self._last_direction: np.ndarray | None = None

    def support(self, direction) -> ConicPoint:
        """Maximize ``direction · (prob, loss)`` over feasible cell moments."""
        vector = np.asarray(direction, dtype=float).reshape(-1)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"support direction must be a finite 2-vector: {vector}")
        scale = max(float(np.max(np.abs(vector))), 1.0)
        normalized = vector / scale
        self.direction.value = normalized
        warm_start = bool(
            self.settings.warm_start
            and self._last_direction is not None
            and np.allclose(self._last_direction, normalized)
        )
        outcome = solve_with_rescue(
            self.support_problem,
            self.constraints,
            self.settings,
            warm_start=warm_start,
        )
        self._last_direction = np.array(normalized, copy=True)
        return self._point_from_outcome(outcome, normalized, scale=scale)

    def feasibility(self) -> ConicPoint:
        """Feasibility certificate: support in the zero direction."""
        return self.support((0.0, 0.0))

    def bounds(self, target: str) -> tuple[float, float, float]:
        """Marginal interval ``(lo, hi, violation)`` for ``prob`` or ``loss``."""
        if target not in _TARGETS:
            raise ValueError(f"unknown target {target!r}; expected {_TARGETS}")
        axis = 0 if target == "prob" else 1
        unit = np.zeros(2, dtype=float)
        unit[axis] = 1.0
        high = self.support(unit)
        low = self.support(-unit)
        if not high.ok or not low.ok:
            return float("nan"), float("nan"), float("inf")
        raw_hi = float(high.value)
        raw_lo = -float(low.value)
        logical_upper = 1.0 if target == "prob" else self.K
        if (
            raw_lo < -_BOUND_LOWER_TOLERANCE
            or raw_hi > logical_upper + _BOUND_UPPER_TOLERANCE
            or raw_lo > raw_hi + _BOUND_ORDER_TOLERANCE
        ):
            return float("nan"), float("nan"), float("inf")
        return (
            max(0.0, raw_lo),
            min(logical_upper, raw_hi),
            max(float(low.violation), float(high.violation)),
        )

    def depth_bounds(
        self, minimum_prob_upper: float = _DEFAULT_MINIMUM_PROB_UPPER
    ) -> tuple[float, float, float, bool]:
        """Conditional-depth interval under the first-moment crash normalization ``λ = 1/A₁``."""
        prob_lo, prob_hi, prob_violation = self.bounds("prob")
        if not np.isfinite([prob_hi, prob_violation]).all():
            return float("nan"), float("nan"), float("inf"), False
        if float(prob_hi) < float(minimum_prob_upper):
            return float("nan"), float("nan"), float(prob_violation), False

        mass = cp.Variable(self.n_cells, nonneg=True, name="crash_mass")
        first = cp.Variable(self.n_cells, nonneg=True, name="crash_first_moment")
        second = cp.Variable(
            self.threshold_cell,
            nonneg=True,
            name="crash_below_second_moment",
        )
        atom = cp.Variable(nonneg=True, name="crash_mass_threshold_atom")
        lam = cp.Variable(nonneg=True, name="inverse_crash_first_moment")
        constraints: list[cp.Constraint] = [
            cp.sum(mass) == lam,
            cp.sum(first) == lam,
        ]
        constraints.extend(self._linear_cell_constraints(mass, first, atom))
        constraints.extend(self._second_moment_constraints(mass, first, second))
        strikes = self.quotes["x"].to_numpy(dtype=float)
        lower = self.quotes["norm_l"].to_numpy(dtype=float)
        upper = self.quotes["norm_u"].to_numpy(dtype=float)
        for strike, bid, ask in zip(strikes, lower, upper):
            payoff = self._call_payoff(mass, first, float(strike))
            constraints.extend(
                [
                    payoff >= float(bid) * lam,
                    payoff <= float(ask) * lam,
                ]
            )
        below_first = cp.sum(first[: self.threshold_cell])
        below_second = cp.sum(second)
        constraints.append(below_first + self.K * atom == 1.0)
        depth = self.K * below_first - below_second
        constraints.extend([depth >= 0.0, depth <= self.K])
        low = self._depth_optimum("min", constraints, depth)
        high = self._depth_optimum("max", constraints, depth)
        violation = max(float(prob_violation), float(low.violation), float(high.violation))
        if not low.ok or not high.ok:
            return float("nan"), float("nan"), violation, False
        minimum = float(low.value)
        maximum = float(high.value)
        if (
            minimum < -_BOUND_LOWER_TOLERANCE
            or maximum > self.K + _BOUND_UPPER_TOLERANCE
            or minimum > maximum + _BOUND_ORDER_TOLERANCE
        ):
            return float("nan"), float("nan"), violation, False
        return max(0.0, minimum), min(self.K, maximum), violation, True

    def interior_margin(self) -> tuple[bool, float, float, str]:
        """Largest uniform fractional quote slack ``η`` on the linear cell system."""
        widths = (
            self.quotes["norm_u"].to_numpy(dtype=float)
            - self.quotes["norm_l"].to_numpy(dtype=float)
        )
        if np.any(widths <= _ZERO_WIDTH_TOLERANCE):
            return False, 0.0, 0.0, "zero_width_quote"
        margin = cp.Variable(nonneg=True, name="quote_margin")
        constraints = list(self.linear_structural_constraints)
        constraints.append(margin <= _ETA_UPPER)
        lower = self.quotes["norm_l"].to_numpy(dtype=float)
        upper = self.quotes["norm_u"].to_numpy(dtype=float)
        for payoff, bid, ask, width in zip(self.quote_calls, lower, upper, widths):
            constraints.extend(
                [
                    payoff >= float(bid) + float(width) * margin,
                    payoff <= float(ask) - float(width) * margin,
                ]
            )
        problem = cp.Problem(cp.Maximize(margin), constraints)
        outcome = solve_with_rescue(
            problem,
            constraints,
            self.settings,
            warm_start=False,
        )
        value = float(outcome.value) if np.isfinite(outcome.value) else float("nan")
        ok = bool(
            outcome.ok
            and np.isfinite(value)
            and value > _INTERIOR_MARGIN_CUTOFF
            and outcome.violation <= self.settings.violation_tolerance
        )
        return ok, value, float(outcome.violation), str(outcome.status)

    def scenario_feasible(self, prob: float, loss: float) -> bool:
        """Return whether some feasible measure attains ``(prob, loss)``."""
        pinned = list(self.constraints) + [
            self.target == np.array([float(prob), float(loss)], dtype=float)
        ]
        problem = cp.Problem(cp.Minimize(0.0 * cp.sum(self.z)), pinned)
        outcome = solve_with_rescue(
            problem,
            pinned,
            self.settings,
            warm_start=False,
        )
        if outcome.ok:
            return True
        if outcome.status in _CERTIFIED_INFEASIBLE:
            return False
        raise SolverFailure(f"scenario feasibility was not certified: {outcome.status}")

    def oracle(self, normalizer: BoxNormalizer) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
        """Unit-box support oracle ``u -> (x(u), h̄(u))`` for hull/grid reconstruction."""

        def query(unit_direction) -> tuple[np.ndarray, float]:
            direction = np.asarray(unit_direction, dtype=float).reshape(2)
            raw = normalizer.raw_direction(direction)
            result = self.support(raw)
            if not result.ok:
                raise SolverFailure(
                    f"support failed for direction {raw[0]}, {raw[1]}: {result.status}"
                )
            return (
                normalizer.to_unit(result.point),
                normalizer.unit_support(float(result.value), direction),
            )

        return query

    def _right_endpoint(self, cell: int) -> float | None:
        if cell + 1 < len(self.boundaries):
            return float(self.boundaries[cell + 1])
        return None

    def _call_payoff(
        self, mass: cp.Variable, first: cp.Variable, strike: float
    ) -> cp.Expression:
        active = self.cell_starts >= float(strike) - _CELL_STRIKE_TOLERANCE
        if not np.any(active):
            return 0.0 * mass[0]
        start = int(np.argmax(active))
        return cp.sum(first[start:] - float(strike) * mass[start:])

    def _linear_cell_constraints(
        self, mass: cp.Variable, first: cp.Variable, atom: cp.Variable
    ) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for cell, left in enumerate(self.cell_starts):
            constraints.append(first[cell] >= float(left) * mass[cell])
            right = self._right_endpoint(cell)
            if right is not None:
                constraints.append(first[cell] <= float(right) * mass[cell])
        threshold = self.threshold_cell
        constraints.append(atom <= mass[threshold])
        right = self._right_endpoint(threshold)
        if right is not None:
            constraints.append(
                first[threshold]
                <= float(right) * mass[threshold] - (float(right) - self.K) * atom
            )
        return constraints

    def _second_moment_constraints(
        self, mass: cp.Variable, first: cp.Variable, second: cp.Variable
    ) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for cell in range(self.threshold_cell):
            left = float(self.cell_starts[cell])
            right = self._right_endpoint(cell)
            constraints.append(
                cp.SOC(
                    mass[cell] + second[cell],
                    cp.hstack([2.0 * first[cell], mass[cell] - second[cell]]),
                )
            )
            if right is not None:
                constraints.append(
                    second[cell]
                    <= (left + float(right)) * first[cell]
                    - left * float(right) * mass[cell]
                )
        return constraints

    def _depth_optimum(
        self, sense: str, constraints: list[cp.Constraint], depth: cp.Expression
    ) -> ConicOutcome:
        if sense == "min":
            problem = cp.Problem(
                cp.Minimize(_DEPTH_OBJECTIVE_SCALE * depth), constraints
            )
        elif sense == "max":
            problem = cp.Problem(
                cp.Maximize(_DEPTH_OBJECTIVE_SCALE * depth), constraints
            )
        else:
            raise ValueError(f"unknown depth sense {sense!r}")
        outcome = solve_with_rescue(
            problem,
            constraints,
            self.settings,
            warm_start=False,
        )
        if outcome.ok:
            return ConicOutcome(
                ok=True,
                value=float(outcome.value) / _DEPTH_OBJECTIVE_SCALE,
                status=str(outcome.status),
                solver=str(outcome.solver),
                violation=float(outcome.violation),
                iterations=outcome.iterations,
                solve_time=outcome.solve_time,
            )
        return self._bisect_depth(sense, constraints, depth)

    def _bisect_depth(
        self, sense: str, base: list[cp.Constraint], depth: cp.Expression
    ) -> ConicOutcome:
        # Direct min/max can hit Clarabel InsufficientProgress when the
        # optimum sits on the boundary depth = 0; feasibility of depth ≷ t
        # stays well posed. Endpoints 0 and K are feasible whenever the base
        # system is, and a loose residual must not push the bracket outside them.
        def probe(mid: float) -> tuple[ConicOutcome, float | None]:
            extra = depth >= mid if sense == "max" else depth <= mid
            pinned = list(base) + [extra]
            problem = cp.Problem(cp.Minimize(0.0 * depth), pinned)
            outcome = solve_with_rescue(
                problem,
                pinned,
                self.settings,
                warm_start=False,
            )
            return outcome, finite_expression(depth)

        def certify_endpoint() -> ConicOutcome:
            problem = cp.Problem(cp.Minimize(0.0 * depth), list(base))
            outcome = solve_with_rescue(
                problem,
                list(base),
                self.settings,
                warm_start=False,
            )
            if outcome.ok:
                return outcome
            auxiliary = replace(
                self.settings,
                solver="SCIPY",
                fallbacks=(),
                warm_start=False,
            )
            rescued = solve_with_rescue(
                problem,
                list(base),
                auxiliary,
                warm_start=False,
            )
            return rescued if rescued.ok else outcome

        return bisect_feasible_bound(
            sense=sense,
            upper=float(self.K),
            probe=probe,
            certify_endpoint=certify_endpoint,
            solver=str(self.settings.solver),
        )

    def _point_from_outcome(
        self, outcome, direction: np.ndarray, *, scale: float = 1.0
    ) -> ConicPoint:
        missing = np.full(2, np.nan)
        values = (
            np.asarray(self.target.value, dtype=float).reshape(2)
            if self.target.value is not None
            else missing.copy()
        )
        ok = bool(outcome.ok and np.all(np.isfinite(values)))
        value = float(outcome.value) if ok and np.isfinite(outcome.value) else float("nan")
        if ok:
            expected = float(direction @ values)
            residual = abs(expected - float(outcome.value))
            limit = self.settings.objective_tolerance * (1.0 + abs(float(outcome.value)))
            if residual > limit:
                ok = False
                value = float("nan")
                values = missing.copy()
            else:
                value = float(outcome.value) * float(scale)
        return ConicPoint(
            ok=ok,
            point=values if ok else missing.copy(),
            value=value,
            status=str(outcome.status),
            solver=str(outcome.solver),
            violation=float(outcome.violation),
        )
