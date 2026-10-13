from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from crashfrontier.errors import SolverFailure
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.physical.cones import moment_cone_order2, moment_cone_order3
from crashfrontier.physical.solve import (
    ConicOutcome,
    ConicSettings,
    bisect_feasible_bound,
    finite_expression,
    solve_with_rescue,
)

__all__ = ["ConicPoint", "Gamma2Model", "RegularityWitness"]

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
_DEFAULT_MINIMUM_PROB_UPPER = 1e-4
_M2_T_FLOOR = 1e-12
_DEPTH_OBJECTIVE_SCALE = 100.0
_CERTIFIED_INFEASIBLE = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
_TARGETS = ("prob", "loss")


@dataclass(frozen=True)
class ConicPoint:
    """Support value and ``(prob, loss)`` witness of one conic solve."""

    ok: bool
    point: np.ndarray
    value: float
    status: str
    solver: str
    violation: float


@dataclass(frozen=True)
class RegularityWitness:
    """Sufficient interior certificate for the γ=2 closure assumption.

    ``quote_slack`` is the strict call-price gap. ``left_mass`` and
    ``right_mass`` are probability on the open intervals to the left and right
    of the forward. ``margin`` is the smallest of those three quantities.
    A failed status is inconclusive: it does not prove the assumption false.
    """

    success: bool
    margin: float
    quote_slack: float
    left_mass: float
    right_mass: float
    violation: float
    status: str
    solver: str
    atom_locations: np.ndarray
    atom_masses: np.ndarray


class Gamma2Model:
    """γ=2 physical crash model with cell moment cones and a threshold atom.

    A finite ``support_cap`` is the compact model on ``[0, B]``.
    ``support_cap=None`` leaves the last cell unbounded. The boundary
    ``t = 0`` is the homogeneous closure, not a finite-second-moment probability.
    """

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
        self.y = cp.Variable((4, n_cells), name="moments")
        self.t = cp.Variable(nonneg=True, name="inverse_second_moment")
        self.atom = cp.Variable(nonneg=True, name="threshold_atom")

        structural = [
            cp.sum(self.y[0, :]) == self.t,
            cp.sum(self.y[1, :]) == self.t,
            cp.sum(self.y[2, :]) == 1.0,
        ]
        structural.extend(self._cell_constraints(self.y, self.atom))
        self.structural_constraints = list(structural)

        self.quote_calls: list[cp.Expression] = []
        self.quote_constraints: list[cp.Constraint] = []
        strikes = panel["x"].to_numpy(dtype=float)
        lower = panel["norm_l"].to_numpy(dtype=float)
        upper = panel["norm_u"].to_numpy(dtype=float)
        for strike, bid, ask in zip(strikes, lower, upper):
            payoff = self._call_payoff(self.y, float(strike))
            self.quote_calls.append(payoff)
            self.quote_constraints.extend(
                [
                    payoff >= float(bid) * self.t,
                    payoff <= float(ask) * self.t,
                ]
            )
        constraints = structural + self.quote_constraints
        below_second = cp.sum(self.y[2, : self.threshold_cell])
        below_third = cp.sum(self.y[3, : self.threshold_cell])
        probability = below_second + (self.K**2) * self.atom
        loss = self.K * below_second - below_third
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
        self.direction = cp.Parameter(2, name="gamma2_support_direction")
        self.support_problem = cp.Problem(
            cp.Maximize(self.direction @ self.target),
            self.constraints,
        )
        self._last_direction: np.ndarray | None = None

    def support(self, direction) -> ConicPoint:
        """Maximize ``direction · (prob, loss)`` over feasible scaled moments."""
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
        """Conditional-depth interval under the second normalization ``λ = 1/A₂``."""
        prob_lo, prob_hi, prob_violation = self.bounds("prob")
        if not np.isfinite([prob_hi, prob_violation]).all():
            return float("nan"), float("nan"), float("inf"), False
        if float(prob_hi) < float(minimum_prob_upper):
            return float("nan"), float("nan"), float(prob_violation), False

        moments = cp.Variable((4, self.n_cells), name="crash_mass_moments")
        lam = cp.Variable(nonneg=True, name="inverse_crash_second_moment")
        atom = cp.Variable(nonneg=True, name="crash_mass_threshold_atom")
        constraints: list[cp.Constraint] = [
            cp.sum(moments[0, :]) == lam,
            cp.sum(moments[1, :]) == lam,
        ]
        constraints.extend(self._cell_constraints(moments, atom))
        strikes = self.quotes["x"].to_numpy(dtype=float)
        lower = self.quotes["norm_l"].to_numpy(dtype=float)
        upper = self.quotes["norm_u"].to_numpy(dtype=float)
        for strike, bid, ask in zip(strikes, lower, upper):
            payoff = self._call_payoff(moments, float(strike))
            constraints.extend(
                [
                    payoff >= float(bid) * lam,
                    payoff <= float(ask) * lam,
                ]
            )
        below_second = cp.sum(moments[2, : self.threshold_cell])
        below_third = cp.sum(moments[3, : self.threshold_cell])
        constraints.append(below_second + (self.K**2) * atom == 1.0)
        depth = self.K * below_second - below_third
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
        """Largest uniform quote slack ``margin`` in the scaled call identities."""
        margin = cp.Variable(nonneg=True, name="quote_margin")
        constraints = list(self.structural_constraints)
        lower = self.quotes["norm_l"].to_numpy(dtype=float)
        upper = self.quotes["norm_u"].to_numpy(dtype=float)
        for payoff, bid, ask in zip(self.quote_calls, lower, upper):
            constraints.extend(
                [
                    payoff >= float(bid) * self.t + margin,
                    payoff <= float(ask) * self.t - margin,
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

    def m2_bounds(self) -> tuple[float, float, float]:
        """Interval ``(lo, hi, violation)`` for the physical second moment ``1/t``."""
        lower_problem = cp.Problem(cp.Minimize(self.t), self.constraints)
        upper_problem = cp.Problem(cp.Maximize(self.t), self.constraints)
        lower = solve_with_rescue(
            lower_problem,
            self.constraints,
            self.settings,
            warm_start=False,
        )
        upper = solve_with_rescue(
            upper_problem,
            self.constraints,
            self.settings,
            warm_start=False,
        )
        violation = max(float(lower.violation), float(upper.violation))
        if (
            not lower.ok
            or not upper.ok
            or not np.isfinite(lower.value)
            or not np.isfinite(upper.value)
            or float(upper.value) <= 0.0
        ):
            return float("nan"), float("nan"), float("inf")
        m2_lower = 1.0 / float(upper.value)
        t_lower = float(lower.value)
        closure_floor = max(
            _M2_T_FLOOR,
            10.0 * float(lower.violation),
            10.0 * float(self.settings.tolerance),
        )
        if self.support_cap is None and t_lower <= closure_floor:
            m2_upper = float("inf")
        elif t_lower <= _M2_T_FLOOR:
            return float("nan"), float("nan"), float("inf")
        else:
            m2_upper = 1.0 / t_lower
        if np.isfinite(m2_upper) and m2_lower > m2_upper + _BOUND_ORDER_TOLERANCE:
            return float("nan"), float("nan"), float("inf")
        return m2_lower, m2_upper, violation

    def regularity_sufficient_witness(self) -> RegularityWitness:
        """Finite-atomic sufficient certificate for interior regularity.

        Candidate atoms sit strictly inside cells cut by the strikes, the
        threshold, zero, one, and the support cap. A positive result certifies
        strict quote slack and positive mass on bounded open intervals on both
        sides of the forward. Failure is only inconclusive.
        ``interior_margin`` remains the scaled quote-slack program.
        """
        cap = (
            self.support_cap
            if self.support_cap is not None
            else max(
                2.0,
                2.0
                * float(
                    max(
                        1.0,
                        float(self.quotes["x"].max()),
                        self.K,
                    )
                ),
            )
        )
        forbidden = np.unique(
            np.round(
                np.r_[
                    0.0,
                    1.0,
                    self.K,
                    self.quotes["x"].to_numpy(dtype=float),
                    cap,
                ],
                12,
            )
        )
        forbidden = forbidden[forbidden >= 0.0]
        locations = 0.5 * (forbidden[:-1] + forbidden[1:])
        left = locations < 1.0
        right = locations > 1.0
        empty = np.empty(0, dtype=float)
        if not left.any() or not right.any():
            return self._witness(
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                float("inf"),
                "inconclusive:no_open_intervals",
                locations,
                empty,
            )

        count = len(locations)
        t_index = count
        margin_index = count + 1
        variables = count + 2
        objective = np.zeros(variables)
        objective[margin_index] = -1.0
        equality = np.zeros((3, variables))
        equality[0, :count] = 1.0
        equality[0, t_index] = -1.0
        equality[1, :count] = locations
        equality[1, t_index] = -1.0
        equality[2, :count] = locations**2
        equality_rhs = np.array([0.0, 0.0, 1.0])
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for quote in self.quotes.itertuples(index=False):
            payoff = np.maximum(locations - float(quote.x), 0.0)
            lower_row = np.zeros(variables)
            lower_row[:count] = -payoff
            lower_row[t_index] = float(quote.norm_l)
            lower_row[margin_index] = 1.0
            rows.append(lower_row)
            rhs.append(0.0)
            upper_row = np.zeros(variables)
            upper_row[:count] = payoff
            upper_row[t_index] = -float(quote.norm_u)
            upper_row[margin_index] = 1.0
            rows.append(upper_row)
            rhs.append(0.0)
        for mask in (left, right):
            row = np.zeros(variables)
            row[:count][mask] = -1.0
            row[margin_index] = 1.0
            rows.append(row)
            rhs.append(0.0)
        result = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            A_eq=equality,
            b_eq=equality_rhs,
            bounds=[(0.0, None)] * variables,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "ipm_optimality_tolerance": 1e-10,
            },
        )
        if not result.success or result.x is None or not np.isfinite(result.x).all():
            return self._witness(
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                float("inf"),
                f"inconclusive:{result.message}",
                locations,
                empty,
            )
        inequality_residual = np.asarray(result.ineqlin.residual, dtype=float)
        equality_residual = np.asarray(result.eqlin.residual, dtype=float)
        violation = max(
            float(np.maximum(-inequality_residual, 0.0).max(initial=0.0)),
            float(np.abs(equality_residual).max(initial=0.0)),
        )
        weights = np.asarray(result.x[:count], dtype=float)
        t_value = float(result.x[t_index])
        if not np.isfinite(t_value) or t_value <= 0.0:
            return self._witness(
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                violation,
                "inconclusive:invalid_normalization",
                locations,
                empty,
            )
        probabilities = weights / t_value
        calls = np.array(
            [
                float(np.dot(probabilities, np.maximum(locations - float(quote.x), 0.0)))
                for quote in self.quotes.itertuples(index=False)
            ]
        )
        quote_slack = float(
            min(
                np.min(calls - self.quotes["norm_l"].to_numpy(dtype=float)),
                np.min(self.quotes["norm_u"].to_numpy(dtype=float) - calls),
            )
        )
        left_mass = float(probabilities[left].sum())
        right_mass = float(probabilities[right].sum())
        margin = min(quote_slack, left_mass, right_mass)
        success = bool(margin > 1e-9 and violation <= 1e-8)
        return self._witness(
            success,
            margin,
            quote_slack,
            left_mass,
            right_mass,
            violation,
            "certified" if success else "inconclusive:non_strict_margin",
            locations,
            probabilities,
        )

    @staticmethod
    def _witness(
        success: bool,
        margin: float,
        quote_slack: float,
        left_mass: float,
        right_mass: float,
        violation: float,
        status: str,
        locations: np.ndarray,
        masses: np.ndarray,
    ) -> RegularityWitness:
        return RegularityWitness(
            success=success,
            margin=float(margin),
            quote_slack=float(quote_slack),
            left_mass=float(left_mass),
            right_mass=float(right_mass),
            violation=float(violation),
            status=status,
            solver="HIGHS",
            atom_locations=locations,
            atom_masses=masses,
        )

    def scenario_feasible(self, prob: float, loss: float) -> bool:
        """Return whether some feasible measure attains ``(prob, loss)``."""
        pinned = list(self.constraints) + [
            self.target == np.array([float(prob), float(loss)], dtype=float)
        ]
        problem = cp.Problem(cp.Minimize(0.0 * self.t), pinned)
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

    def _call_payoff(self, moments: cp.Variable, strike: float) -> cp.Expression:
        active = self.cell_starts >= float(strike) - _CELL_STRIKE_TOLERANCE
        if not np.any(active):
            return 0.0 * moments[0, 0]
        start = int(np.argmax(active))
        return cp.sum(moments[1, start:] - float(strike) * moments[0, start:])

    def _cell_constraints(self, moments: cp.Variable, atom: cp.Variable) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for cell, left in enumerate(self.cell_starts):
            right = self._right_endpoint(cell)
            if cell < self.threshold_cell:
                constraints.extend(
                    moment_cone_order3(
                        moments[0, cell],
                        moments[1, cell],
                        moments[2, cell],
                        moments[3, cell],
                        float(left),
                        right,
                    )
                )
                continue
            orders = [moments[order, cell] for order in range(3)]
            if cell == self.threshold_cell:
                orders = [orders[power] - atom * (self.K**power) for power in range(3)]
            constraints.append(moments[3, cell] == 0.0)
            constraints.extend(
                moment_cone_order2(orders[0], orders[1], orders[2], float(left), right)
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
