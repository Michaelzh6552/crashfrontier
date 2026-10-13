from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crashfrontier.config import (
    DEFAULT_P0,
    DEFAULT_SUPPORT_CAP,
    DEFAULT_THRESHOLD,
    ConicSettings,
    GridSettings,
    LPSettings,
)
from crashfrontier.data.cross_section import CrossSection
from crashfrontier.data.quote_sets import select_quote_set
from crashfrontier.errors import InfeasibleCrossSectionError, SolverFailure
from crashfrontier.physical.gamma1 import Gamma1Model
from crashfrontier.physical.gamma2 import Gamma2Model
from crashfrontier.geometry.depth import (
    area_ratio,
    depth_boundary as sample_depth_boundary,
    marginal_wedge,
    transformed_area,
)
from crashfrontier.geometry.grid import enumerate_by_directions
from crashfrontier.geometry.hull import enumerate_hull
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.geometry.polygon import (
    affine_dimension,
    clip_polygon,
    convex_hull,
    halfplane_intersection,
    point_in_convex_polygon,
    segment_endpoints,
    unique_points,
)
from crashfrontier.riskneutral.certificates import DualCertificate, dual_certificate, interior_margin
from crashfrontier.riskneutral.lp import (
    ConstraintSystem,
    MeasureProblem,
    build_problem,
    configure_constraints,
    impose_support_cap,
    marginal_bounds,
    portfolio_bounds,
    scenario_feasible,
    solve_lp,
    support,
)

__all__ = [
    "Frontier",
    "binding_quotes",
    "compute_frontier",
    "nested_frontiers",
    "portfolio_bound",
    "scenario_check",
]

logger = logging.getLogger("crashfrontier")

_SUMMARY_KEYS = (
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
)
_AXIS_DIRECTIONS = (
    np.array([1.0, 0.0], dtype=float),
    np.array([-1.0, 0.0], dtype=float),
    np.array([0.0, 1.0], dtype=float),
    np.array([0.0, -1.0], dtype=float),
)
_RELAXED_WARNING = (
    "quotes admit no arbitrage-free measure; using minimum slack budget δ*"
)
_BINDING_RELAXED_WARNING = (
    "hard system is infeasible; binding quotes use the relaxed slack-budget system"
)
_PHYSICAL_INTERIOR_WARNING = (
    "interior_margin does not hold; continuing with the physical model"
)
_SHARED_NORMALIZER_RETRY_WARNING = (
    "support reconstruction for quote_set=%s failed under the shared put_pair "
    "normalizer; retrying with this set's own marginal box"
)
_INFEASIBLE_HINT = (
    " The default physical quote_set is 'full_otm'; you can retry with "
    "quote_set='put_wing' or support_cap=None (neither is guaranteed to restore "
    "feasibility)."
)
_CUSTOM_SET_NAME = "custom"
_PHYSICAL_GAMMAS = {1, 2}
_HALFPLANE_COLUMNS = ("a_prob", "a_loss", "h")
_VERTEX_COLUMNS = ("kind", "order", "prob", "loss", "depth")


@dataclass
class Frontier:
    """Joint crash-probability / expected-loss set for one cross-section."""

    date: pd.Timestamp
    expiry: pd.Timestamp
    days: int
    am_settlement: int
    K: float
    gamma: int
    support_cap: float | None
    quote_set: str
    n_quotes: int
    n_strikes: int
    forward: float
    forward_source: str
    rate: float
    discount: float
    spot: float
    mode: str
    method: str
    status: str
    dimension: int
    inner: np.ndarray
    outer: np.ndarray
    halfplanes: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    depth_lo: float
    depth_hi: float
    p0: float
    area: float
    area_outer: float
    baseline_area: float
    area_ratio: float
    interior_ok: bool
    interior_margin: float
    diagnostics: dict[str, object]
    normalizer: BoxNormalizer
    quotes: pd.DataFrame

    def depth_boundary(
        self,
        samples_per_edge: int = 80,
        *,
        p_floor: float | None = None,
    ) -> np.ndarray:
        """Inner-set image under ``(p, L) -> (p, L/p)`` above a probability floor."""
        floor = self.p0 if p_floor is None else float(p_floor)
        return sample_depth_boundary(self.inner, samples_per_edge, p0=floor)

    def contains(self, prob, loss) -> str:
        """Classify ``(prob, loss)`` as ``inside``, ``outside``, or ``boundary``.

        ``inside`` means the point lies in the inner polygon (including its
        boundary). ``outside`` means it lies strictly outside the outer
        polygon. The remaining shell, including the outer boundary when it is
        distinct from the inner set, is ``boundary``.
        """
        point = np.array([float(prob), float(loss)], dtype=float)
        if point_in_convex_polygon(point, self.inner):
            return "inside"
        if not point_in_convex_polygon(point, self.outer):
            return "outside"
        return "boundary"

    def summary(self) -> dict[str, object]:
        """One-row scalar record in the fixed export column order."""
        return {
            "date": self.date,
            "expiry": self.expiry,
            "days": int(self.days),
            "am_settlement": int(self.am_settlement),
            "K": float(self.K),
            "gamma": int(self.gamma),
            "support_cap": None if self.support_cap is None else float(self.support_cap),
            "quote_set": str(self.quote_set),
            "n_quotes": int(self.n_quotes),
            "n_strikes": int(self.n_strikes),
            "forward": float(self.forward),
            "forward_source": str(self.forward_source),
            "rate": float(self.rate),
            "discount": float(self.discount),
            "spot": float(self.spot),
            "mode": str(self.mode),
            "method": str(self.method),
            "status": str(self.status),
            "dimension": int(self.dimension),
            "prob_lo": float(self.lo[0]),
            "prob_hi": float(self.hi[0]),
            "loss_lo": float(self.lo[1]),
            "loss_hi": float(self.hi[1]),
            "depth_lo": float(self.depth_lo),
            "depth_hi": float(self.depth_hi),
            "area": float(self.area),
            "area_outer": float(self.area_outer),
            "baseline_area": float(self.baseline_area),
            "area_ratio": float(self.area_ratio),
            "interior_ok": bool(self.interior_ok),
            "interior_margin": float(self.interior_margin),
            "support_calls": int(self.diagnostics.get("support_calls", 0)),
            "max_gap": _as_float(self.diagnostics.get("max_gap")),
            "solver_gap": _as_float(
                self.diagnostics.get("max_duality_gap")
                if int(self.gamma) == 0
                else self.diagnostics.get("max_violation")
            ),
            "elapsed_seconds": _as_float(self.diagnostics.get("elapsed_seconds")),
        }

    def vertices(self) -> pd.DataFrame:
        """Inner and outer vertices with ``depth = loss / prob`` when ``prob > 0``."""
        records: list[dict[str, object]] = []
        for kind, cloud in (("inner", self.inner), ("outer", self.outer)):
            points = _as_cloud(cloud)
            for order, point in enumerate(points):
                prob = float(point[0])
                loss = float(point[1])
                records.append(
                    {
                        "kind": kind,
                        "order": int(order),
                        "prob": prob,
                        "loss": loss,
                        "depth": loss / prob if prob > 0.0 else float("nan"),
                    }
                )
        return pd.DataFrame.from_records(records, columns=list(_VERTEX_COLUMNS))

    def halfplane_table(self) -> pd.DataFrame:
        """Support half-planes ``a_prob * p + a_loss * L <= h`` in raw coordinates."""
        planes = np.asarray(self.halfplanes, dtype=float)
        if planes.size == 0:
            return pd.DataFrame(columns=list(_HALFPLANE_COLUMNS))
        return pd.DataFrame(planes.reshape(-1, 3), columns=list(_HALFPLANE_COLUMNS))

    def to_dict(self) -> dict:
        """Strict-JSON record: arrays as lists, dates as text, non-finite values as null."""
        return _jsonify({
            "date": _iso_date(self.date),
            "expiry": _iso_date(self.expiry),
            "days": int(self.days),
            "am_settlement": int(self.am_settlement),
            "K": float(self.K),
            "gamma": int(self.gamma),
            "support_cap": None if self.support_cap is None else float(self.support_cap),
            "quote_set": str(self.quote_set),
            "n_quotes": int(self.n_quotes),
            "n_strikes": int(self.n_strikes),
            "forward": float(self.forward),
            "forward_source": str(self.forward_source),
            "rate": float(self.rate),
            "discount": float(self.discount),
            "spot": float(self.spot),
            "mode": str(self.mode),
            "method": str(self.method),
            "status": str(self.status),
            "dimension": int(self.dimension),
            "inner": _as_cloud(self.inner).tolist(),
            "outer": _as_cloud(self.outer).tolist(),
            "halfplanes": _as_halfplanes(self.halfplanes).tolist(),
            "lo": np.asarray(self.lo, dtype=float).reshape(2).tolist(),
            "hi": np.asarray(self.hi, dtype=float).reshape(2).tolist(),
            "depth_lo": float(self.depth_lo),
            "depth_hi": float(self.depth_hi),
            "p0": float(self.p0),
            "area": float(self.area),
            "area_outer": float(self.area_outer),
            "baseline_area": float(self.baseline_area),
            "area_ratio": float(self.area_ratio),
            "interior_ok": bool(self.interior_ok),
            "interior_margin": float(self.interior_margin),
            "diagnostics": _jsonify(self.diagnostics),
            "normalizer": {
                "origin": np.asarray(self.normalizer.origin, dtype=float).reshape(2).tolist(),
                "scale": np.asarray(self.normalizer.scale, dtype=float).reshape(2).tolist(),
            },
            "quotes": _jsonify(self.quotes.to_dict(orient="records")),
        })

    def describe(self) -> str:
        """Multi-line text summary for interactive and CLI display."""
        cap = "none" if self.support_cap is None else f"{float(self.support_cap):g}"
        return "\n".join(
            [
                (
                    f"{_iso_date(self.date)} → {_iso_date(self.expiry)} ({self.days}d)  "
                    f"K={float(self.K):.2f}  γ={int(self.gamma)}"
                ),
                (
                    f"quote_set={self.quote_set}  n_quotes={int(self.n_quotes)}  "
                    f"n_strikes={int(self.n_strikes)}"
                ),
                (
                    f"F={float(self.forward):g} ({self.forward_source})  "
                    f"D={float(self.discount):g}  support_cap={cap}"
                ),
                (
                    f"mode={self.mode}  method={self.method}  "
                    f"status={self.status}  dimension={int(self.dimension)}"
                ),
                (
                    f"prob=[{float(self.lo[0]):.6g}, {float(self.hi[0]):.6g}]  "
                    f"loss=[{float(self.lo[1]):.6g}, {float(self.hi[1]):.6g}]  "
                    f"depth=[{float(self.depth_lo):.6g}, {float(self.depth_hi):.6g}]"
                ),
                (
                    f"area={float(self.area):.6g}  area_outer={float(self.area_outer):.6g}  "
                    f"area_ratio={float(self.area_ratio):.6g}  "
                    f"interior_ok={bool(self.interior_ok)}  "
                    f"interior_margin={float(self.interior_margin):.6g}"
                ),
            ]
        )


def compute_frontier(
    cs: CrossSection,
    K: float = DEFAULT_THRESHOLD,
    *,
    gamma: int = 0,
    quote_set: str | pd.DataFrame | None = None,
    design: str | None = None,
    support_cap: float | None | str = "auto",
    mode: str = "auto",
    budget: float | None = None,
    method: str = "auto",
    p0: float = DEFAULT_P0,
    normalizer: BoxNormalizer | None = None,
    lp_settings: LPSettings | None = None,
    conic_settings: ConicSettings | None = None,
    grid_settings: GridSettings | None = None,
    full_strike_count: int | None = None,
) -> Frontier:
    """Build the crash frontier of a cross-section.

    ``gamma=0`` uses the risk-neutral LP. ``gamma=1`` and ``gamma=2`` use the
    physical SOCP models. ``mode="auto"`` (gamma=0) prefers a hard
    (arbitrage-free) system and falls back to the minimum-slack budget when
    the quotes admit no measure.

    ``support_cap="auto"`` uses ``5.0`` when ``gamma>0`` and unbounded support
    when ``gamma=0``. Pass ``support_cap=None`` for unbounded support at any
    gamma.
    """
    gamma = int(gamma)
    if gamma not in {0, 1, 2}:
        raise ValueError(f"unsupported gamma {gamma}; expected 0, 1, or 2")
    if lp_settings is None:
        lp_settings = LPSettings()
    if grid_settings is None:
        grid_settings = GridSettings()
    if conic_settings is None:
        conic_settings = ConicSettings()
    started = time.perf_counter()
    quotes, set_name, _design = _resolve_quotes(cs, K, quote_set, design, gamma)
    cap = _resolve_support_cap(gamma, support_cap)
    if gamma in _PHYSICAL_GAMMAS:
        return _compute_physical(
            cs,
            K,
            gamma=gamma,
            quotes=quotes,
            set_name=set_name,
            cap=cap,
            method=method,
            p0=p0,
            normalizer=normalizer,
            lp_settings=lp_settings,
            conic_settings=conic_settings,
            grid_settings=grid_settings,
            started=started,
        )
    try:
        problem, system, relaxed_fallback = _configure_gamma0(
            quotes,
            K,
            mode=mode,
            budget=budget,
            support_cap=cap,
            lp_settings=lp_settings,
            full_strike_count=full_strike_count,
            warn_relaxed=True,
        )
    except InfeasibleCrossSectionError as exc:
        raise _infeasible_error(
            cs, set_name, gamma, cap, status="infeasible", detail=str(exc)
        ) from exc
    used = problem.quotes
    interior_ok, eta_star, interior_status = _interior_state(problem, system, lp_settings)
    lo, hi = marginal_bounds(system, lp_settings)
    lo = np.asarray(lo, dtype=float).reshape(2).copy()
    hi = np.asarray(hi, dtype=float).reshape(2).copy()
    reconstruction = method if method != "auto" else "hull"
    if reconstruction not in {"hull", "grid"}:
        raise ValueError(f"unrecognized reconstruction method {method!r}")

    statuses: list[str] = [system.mode, interior_status]
    duality_gaps: list[float] = []
    width = hi - lo
    if np.any(width <= lp_settings.degenerate_tolerance):
        inner, outer, planes, dimension, status, calls, gaps, messages = _degenerate_polygon(
            system, lp_settings
        )
        duality_gaps.extend(gaps)
        statuses.extend(messages)
        statuses.append(status)
        used_method = "degenerate"
        max_gap = 0.0
        box = _box_normalizer(lo, hi, lp_settings.degenerate_tolerance, normalizer)
    else:
        box = normalizer if normalizer is not None else BoxNormalizer(lo, hi - lo)
        oracle = _SupportOracle(system, lp_settings, box)
        if reconstruction == "hull":
            if system.mode == "relaxed":
                predicate = _area_predicate(lp_settings, box, p0)
                max_calls = lp_settings.anytime_max_support_calls
            else:
                predicate = None
                max_calls = lp_settings.max_support_calls
            hull = enumerate_hull(
                oracle,
                edge_tolerance=lp_settings.edge_tolerance,
                point_tolerance=lp_settings.point_tolerance,
                max_calls=max_calls,
                area_predicate=predicate,
            )
            inner = box.to_raw(hull.inner)
            outer = box.to_raw(hull.outer)
            planes = _unit_halfplanes_to_raw(box, hull.certificates)
            dimension = int(hull.dimension)
            status = str(hull.status)
            calls = int(hull.calls)
            max_gap = float(hull.max_gap)
        else:
            grid = enumerate_by_directions(
                oracle,
                levels=grid_settings.levels,
                max_gap=grid_settings.max_normalized_gap,
            )
            inner = box.to_raw(grid.inner)
            outer = box.to_raw(grid.outer)
            planes = _grid_halfplanes_to_raw(box, grid.supports)
            dimension = 2
            status = str(grid.status)
            calls = int(len(grid.supports))
            max_gap = float(grid.max_gap)
        duality_gaps.extend(oracle.duality_gaps)
        statuses.extend(oracle.statuses)
        statuses.append(status)
        used_method = reconstruction

    inner = _clip_to_target_wedge(_as_cloud(inner), float(K))
    outer = _clip_to_target_wedge(_as_cloud(outer), float(K))
    _, baseline = marginal_wedge(lo, hi, float(K), float(p0))
    area, area_outer, ratio = _reported_areas(inner, outer, p0, baseline, dimension)
    depth_lo, depth_hi = _depth_range(inner)
    elapsed = time.perf_counter() - started
    diagnostics: dict[str, object] = {
        "support_calls": int(calls),
        "max_gap": float(max_gap) if np.isfinite(max_gap) else float("nan"),
        "max_duality_gap": max(duality_gaps) if duality_gaps else float("nan"),
        "budget": None if system.budget is None else float(system.budget),
        "minimum_slack": float(system.minimum_slack),
        "statuses": statuses,
        "elapsed_seconds": float(elapsed),
        "relaxed": bool(relaxed_fallback),
    }
    return Frontier(
        date=pd.Timestamp(cs.date),
        expiry=pd.Timestamp(cs.expiry),
        days=int(cs.days),
        am_settlement=int(cs.am_settlement),
        K=float(K),
        gamma=int(gamma),
        support_cap=cap,
        quote_set=set_name,
        n_quotes=int(len(used)),
        n_strikes=int(used["strike"].nunique()) if len(used) else 0,
        forward=float(cs.forward),
        forward_source=str(cs.forward_source),
        rate=float(cs.rate),
        discount=float(cs.discount),
        spot=float(cs.spot),
        mode=str(system.mode),
        method=used_method,
        status=status,
        dimension=int(dimension),
        inner=inner,
        outer=outer,
        halfplanes=planes,
        lo=lo,
        hi=hi,
        depth_lo=depth_lo,
        depth_hi=depth_hi,
        p0=float(p0),
        area=float(area),
        area_outer=float(area_outer),
        baseline_area=float(baseline),
        area_ratio=float(ratio),
        interior_ok=bool(interior_ok),
        interior_margin=float(eta_star),
        diagnostics=diagnostics,
        normalizer=box,
        quotes=used.reset_index(drop=True),
    )


def nested_frontiers(
    cs: CrossSection,
    K: float = DEFAULT_THRESHOLD,
    *,
    gamma: int = 0,
    sets=None,
    accumulate_halfplanes=None,
    **kwargs,
) -> dict[str, Frontier]:
    """Compute nested quote-set frontiers on one cross-section.

    For ``gamma=0`` and ``mode="auto"``, hard feasibility is decided on the
    largest set. If that set is infeasible, every set uses the same relaxed
    slack budget (the largest-set ``δ* + 1e-10``, or the caller-supplied
    ``budget``) and the same ``full_strike_count``.

    For ``gamma>0``, ``accumulate_halfplanes`` defaults to True: each layer's
    outer envelope is the intersection of support half-planes accumulated up
    to that layer. An infeasible layer raises ``InfeasibleCrossSectionError``
    rather than omitting the set.
    """
    gamma = int(gamma)
    names = _nested_names(sets, gamma)
    accumulate = bool(gamma > 0) if accumulate_halfplanes is None else bool(accumulate_halfplanes)
    options = dict(kwargs)
    options.pop("quote_set", None)
    if "support_cap" in options:
        options["support_cap"] = _resolve_support_cap(gamma, options["support_cap"])
    design = options.get("design")
    if design is None:
        design = "riskneutral" if gamma == 0 else "physical"
        options["design"] = design
    largest_quotes = select_quote_set(cs.quotes, K, names[-1], design) if names else cs.quotes
    strike_count = int(largest_quotes["strike"].nunique()) if len(largest_quotes) else 0
    options["full_strike_count"] = strike_count
    mode = options.get("mode", "auto")
    budget = options.get("budget")
    lp_settings = options.get("lp_settings") or LPSettings()
    if gamma == 0 and mode == "auto" and names:
        try:
            _problem, system, fallback = _configure_gamma0(
                largest_quotes,
                K,
                mode="auto",
                budget=budget,
                support_cap=options.get("support_cap"),
                lp_settings=lp_settings,
                full_strike_count=strike_count,
                warn_relaxed=True,
            )
        except InfeasibleCrossSectionError as exc:
            raise _infeasible_error(
                cs,
                names[-1],
                gamma,
                options.get("support_cap"),
                status="infeasible",
                detail=str(exc),
            ) from exc
        options["mode"] = system.mode
        if fallback and budget is None:
            options["budget"] = system.budget
    fronts: dict[str, Frontier] = {}
    accumulated: list[tuple[np.ndarray, float]] = []
    caller_normalizer = options.get("normalizer")
    shared_normalizer = caller_normalizer
    reference_name = "put_pair" if "put_pair" in names else (names[0] if names else None)
    completed: list[str] = []
    for name in names:
        call_options = dict(options)
        if shared_normalizer is not None:
            call_options["normalizer"] = shared_normalizer
        try:
            frontier = compute_frontier(cs, K, gamma=gamma, quote_set=name, **call_options)
        except InfeasibleCrossSectionError as exc:
            done = ", ".join(completed) if completed else "none"
            raise InfeasibleCrossSectionError(
                (
                    f"nested_frontiers failed at quote_set={name!r} "
                    f"(completed: {done}). {exc}"
                ),
                date=exc.date,
                expiry=exc.expiry,
                quote_set=exc.quote_set or name,
                gamma=exc.gamma if exc.gamma is not None else gamma,
                support_cap=exc.support_cap,
                status=exc.status,
            ) from exc
        except SolverFailure:
            retryable = (
                gamma > 0
                and caller_normalizer is None
                and call_options.get("normalizer") is not None
                and name != reference_name
            )
            if not retryable:
                raise
            logger.warning(_SHARED_NORMALIZER_RETRY_WARNING, name)
            retry_options = dict(options)
            retry_options.pop("normalizer", None)
            frontier = compute_frontier(cs, K, gamma=gamma, quote_set=name, **retry_options)
        if (
            gamma > 0
            and caller_normalizer is None
            and shared_normalizer is None
            and name == reference_name
            and frontier.method != "degenerate"
        ):
            width = np.asarray(frontier.hi - frontier.lo, dtype=float).reshape(2)
            if np.all(width > float(lp_settings.degenerate_tolerance)):
                shared_normalizer = BoxNormalizer(
                    np.asarray(frontier.lo, dtype=float).reshape(2).copy(),
                    width.copy(),
                )
        if accumulate:
            new_planes = _halfplane_pairs(frontier.halfplanes)
            frontier = _apply_accumulated_halfplanes(frontier, accumulated)
            accumulated.extend(new_planes)
        fronts[name] = frontier
        completed.append(name)
    return fronts


def scenario_check(
    cs: CrossSection,
    K,
    prob,
    loss,
    *,
    gamma: int = 0,
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> dict:
    """Feasibility of a ``(prob, loss)`` scenario against quotes and geometry."""
    gamma = int(gamma)
    if gamma not in {0, 1, 2}:
        raise ValueError(f"unsupported gamma {gamma}; expected 0, 1, or 2")
    frontier = compute_frontier(cs, float(K), gamma=gamma, quote_set=quote_set, **kwargs)
    geometry = frontier.contains(prob, loss)
    method = "lp" if gamma == 0 else "socp"
    if geometry == "inside":
        feasible = True
    elif geometry == "outside":
        feasible = False
    elif gamma == 0:
        _problem, system, _fallback = _configure_gamma0(
            frontier.quotes,
            float(K),
            mode=frontier.mode,
            budget=frontier.diagnostics.get("budget"),
            support_cap=frontier.support_cap,
            lp_settings=kwargs.get("lp_settings") or LPSettings(),
            full_strike_count=kwargs.get("full_strike_count"),
            warn_relaxed=False,
        )
        feasible = bool(
            scenario_feasible(
                system,
                kwargs.get("lp_settings") or LPSettings(),
                float(prob),
                float(loss),
            )
        )
    else:
        model = _physical_model(
            frontier.quotes,
            float(K),
            gamma,
            frontier.support_cap,
            kwargs.get("conic_settings"),
        )
        feasible = bool(model.scenario_feasible(float(prob), float(loss)))
    return {
        "feasible": bool(feasible),
        "geometry": geometry,
        "method": method,
        "frontier": frontier,
    }


def portfolio_bound(
    cs: CrossSection,
    K,
    u_prob,
    u_loss,
    *,
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> tuple[float, float]:
    """Normalized price interval ``(-h(-u), h(u))`` for a linear crash payoff."""
    gamma = int(kwargs.pop("gamma", 0))
    if gamma != 0:
        raise ValueError("portfolio_bound is defined for gamma=0")
    quotes, name, _design = _resolve_quotes(
        cs, float(K), quote_set, kwargs.get("design"), 0
    )
    cap = _resolve_support_cap(0, kwargs.get("support_cap"))
    try:
        _problem, system, _fallback = _configure_gamma0(
            quotes,
            float(K),
            mode=kwargs.get("mode", "auto"),
            budget=kwargs.get("budget"),
            support_cap=cap,
            lp_settings=kwargs.get("lp_settings", LPSettings()),
            full_strike_count=kwargs.get("full_strike_count"),
            warn_relaxed=True,
        )
    except InfeasibleCrossSectionError as exc:
        raise _infeasible_error(
            cs, name, 0, cap, status="infeasible", detail=str(exc)
        ) from exc
    return portfolio_bounds(
        system,
        kwargs.get("lp_settings", LPSettings()),
        float(u_prob),
        float(u_loss),
    )


def binding_quotes(
    cs: CrossSection,
    K,
    *,
    direction=(1.0, 0.0),
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> DualCertificate:
    """Static super-replicating certificate in a raw ``(prob, loss)`` direction.

    The hard system is used when it is feasible. Otherwise the solver falls
    back to the relaxed slack-budget system and a warning is recorded.
    """
    gamma = int(kwargs.pop("gamma", 0))
    if gamma != 0:
        raise ValueError("binding_quotes is defined for gamma=0")
    quotes, name, _design = _resolve_quotes(
        cs, float(K), quote_set, kwargs.get("design"), 0
    )
    lp_settings = kwargs.get("lp_settings", LPSettings())
    cap = _resolve_support_cap(0, kwargs.get("support_cap"))
    try:
        problem, system, fallback = _configure_gamma0(
            quotes,
            float(K),
            mode="auto",
            budget=kwargs.get("budget"),
            support_cap=cap,
            lp_settings=lp_settings,
            full_strike_count=kwargs.get("full_strike_count"),
            warn_relaxed=False,
        )
    except InfeasibleCrossSectionError as exc:
        raise _infeasible_error(
            cs, name, 0, cap, status="infeasible", detail=str(exc)
        ) from exc
    del problem
    if fallback:
        logger.warning(_BINDING_RELAXED_WARNING)
    vector = np.asarray(direction, dtype=float).reshape(2)
    result = support(system, lp_settings, vector, lexicographic=False)
    if not result.success:
        raise SolverFailure(
            f"support failed for direction {vector[0]}, {vector[1]}: {result.message}"
        )
    return dual_certificate(system, result)


def _compute_physical(
    cs: CrossSection,
    K,
    *,
    gamma: int,
    quotes: pd.DataFrame,
    set_name: str,
    cap: float | None,
    method: str,
    p0: float,
    normalizer: BoxNormalizer | None,
    lp_settings: LPSettings,
    conic_settings: ConicSettings,
    grid_settings: GridSettings,
    started: float,
) -> Frontier:
    reconstruction = method if method != "auto" else "grid"
    if reconstruction not in {"hull", "grid"}:
        raise ValueError(f"unrecognized reconstruction method {method!r}")
    model = _physical_model(quotes, K, gamma, cap, conic_settings)
    used = model.quotes
    interior_ok, eta_star, interior_violation, interior_status = model.interior_margin()
    if not interior_ok:
        logger.warning("%s (status=%s)", _PHYSICAL_INTERIOR_WARNING, interior_status)
    feasible = model.feasibility()
    if not feasible.ok:
        raise _infeasible_error(
            cs,
            set_name,
            gamma,
            cap,
            status=str(feasible.status),
            detail=str(feasible.status),
        )
    prob_lo, prob_hi, prob_violation = model.bounds("prob")
    loss_lo, loss_hi, loss_violation = model.bounds("loss")
    if not np.all(np.isfinite([prob_lo, prob_hi, loss_lo, loss_hi])):
        raise SolverFailure(
            f"physical marginal bounds failed for gamma={gamma}, quote_set={set_name}"
        )
    lo = np.array([float(prob_lo), float(loss_lo)], dtype=float)
    hi = np.array([float(prob_hi), float(loss_hi)], dtype=float)
    depth_lo, depth_hi, depth_violation, depth_ok = model.depth_bounds()
    if not depth_ok:
        depth_lo, depth_hi = float("nan"), float("nan")

    statuses: list[str] = ["socp", str(interior_status), str(feasible.status)]
    violations: list[float] = [
        float(interior_violation),
        float(feasible.violation),
        float(prob_violation),
        float(loss_violation),
    ]
    if np.isfinite(depth_violation):
        violations.append(float(depth_violation))

    width = hi - lo
    if np.any(width <= lp_settings.degenerate_tolerance):
        inner, outer, planes, dimension, status, calls, axis_violations, messages = (
            _degenerate_physical(model, lp_settings)
        )
        violations.extend(axis_violations)
        statuses.extend(messages)
        statuses.append(status)
        used_method = "degenerate"
        max_gap = 0.0
        box = _box_normalizer(lo, hi, lp_settings.degenerate_tolerance, normalizer)
    else:
        box = normalizer if normalizer is not None else BoxNormalizer(lo, hi - lo)
        oracle = _PhysicalOracle(model, box)
        if reconstruction == "hull":
            hull = enumerate_hull(
                oracle,
                edge_tolerance=lp_settings.edge_tolerance,
                point_tolerance=lp_settings.point_tolerance,
                max_calls=lp_settings.max_support_calls,
            )
            inner = box.to_raw(hull.inner)
            outer = box.to_raw(hull.outer)
            planes = _unit_halfplanes_to_raw(box, hull.certificates)
            dimension = int(hull.dimension)
            status = str(hull.status)
            calls = int(hull.calls)
            max_gap = float(hull.max_gap)
        else:
            grid = enumerate_by_directions(
                oracle,
                levels=grid_settings.levels,
                max_gap=grid_settings.max_normalized_gap,
            )
            inner = box.to_raw(grid.inner)
            outer = box.to_raw(grid.outer)
            planes = _grid_halfplanes_to_raw(box, grid.supports)
            dimension = 2
            status = str(grid.status)
            calls = int(len(grid.supports))
            max_gap = float(grid.max_gap)
        violations.extend(oracle.violations)
        statuses.extend(oracle.statuses)
        statuses.append(status)
        used_method = reconstruction

    inner = _clip_to_target_wedge(_as_cloud(inner), float(K))
    outer = _clip_to_target_wedge(_as_cloud(outer), float(K))
    finite_oracle_violations = [
        float(value) for value in violations if np.isfinite(value)
    ]
    depth_probability_floor = max(
        1e-12,
        10.0 * max(finite_oracle_violations, default=0.0),
    )
    inner_depth_lo, inner_depth_hi = _depth_range(
        inner,
        K=float(K),
        probability_floor=depth_probability_floor,
    )
    solver_depth_lo, solver_depth_hi = float(depth_lo), float(depth_hi)
    depth_enclosure_ok = _validate_depth_enclosure(
        depth_lo,
        depth_hi,
        inner_depth_lo,
        inner_depth_hi,
        gamma=gamma,
        quote_set=set_name,
    )
    if (
        depth_enclosure_ok
        and np.isfinite([depth_lo, depth_hi, inner_depth_lo, inner_depth_hi]).all()
    ):
        depth_lo = min(float(depth_lo), float(inner_depth_lo))
        depth_hi = max(float(depth_hi), float(inner_depth_hi))
    elif not depth_enclosure_ok:
        depth_lo = float("nan")
        depth_hi = float("nan")
    _, baseline = marginal_wedge(lo, hi, float(K), float(p0))
    area, area_outer, ratio = _reported_areas(inner, outer, p0, baseline, dimension)
    elapsed = time.perf_counter() - started
    finite_violations = [value for value in violations if np.isfinite(value)]
    diagnostics: dict[str, object] = {
        "support_calls": int(calls),
        "max_gap": float(max_gap) if np.isfinite(max_gap) else float("nan"),
        "max_violation": max(finite_violations) if finite_violations else float("nan"),
        "max_duality_gap": float("nan"),
        "budget": None,
        "minimum_slack": float("nan"),
        "statuses": statuses,
        "elapsed_seconds": float(elapsed),
        "relaxed": False,
        "inner_depth_lo": float(inner_depth_lo),
        "inner_depth_hi": float(inner_depth_hi),
        "inner_depth_probability_floor": float(depth_probability_floor),
        "solver_depth_lo": solver_depth_lo,
        "solver_depth_hi": solver_depth_hi,
        "depth_reconciled": bool(
            depth_enclosure_ok
            and np.isfinite(
                [solver_depth_lo, solver_depth_hi, inner_depth_lo, inner_depth_hi]
            ).all()
            and (
                inner_depth_lo < solver_depth_lo
                or inner_depth_hi > solver_depth_hi
            )
        ),
        "depth_enclosure_ok": bool(depth_enclosure_ok),
        "depth_endpoint_gap": float(
            max(
                abs(solver_depth_lo - float(inner_depth_lo)),
                abs(solver_depth_hi - float(inner_depth_hi)),
            )
        ),
    }
    return Frontier(
        date=pd.Timestamp(cs.date),
        expiry=pd.Timestamp(cs.expiry),
        days=int(cs.days),
        am_settlement=int(cs.am_settlement),
        K=float(K),
        gamma=int(gamma),
        support_cap=cap,
        quote_set=set_name,
        n_quotes=int(len(used)),
        n_strikes=_n_strikes(used),
        forward=float(cs.forward),
        forward_source=str(cs.forward_source),
        rate=float(cs.rate),
        discount=float(cs.discount),
        spot=float(cs.spot),
        mode="socp",
        method=used_method,
        status=status,
        dimension=int(dimension),
        inner=inner,
        outer=outer,
        halfplanes=planes,
        lo=lo,
        hi=hi,
        depth_lo=float(depth_lo),
        depth_hi=float(depth_hi),
        p0=float(p0),
        area=float(area),
        area_outer=float(area_outer),
        baseline_area=float(baseline),
        area_ratio=float(ratio),
        interior_ok=bool(interior_ok),
        interior_margin=float(eta_star) if np.isfinite(eta_star) else float("nan"),
        diagnostics=diagnostics,
        normalizer=box,
        quotes=used.reset_index(drop=True),
    )


def _physical_model(quotes, K, gamma: int, support_cap, settings: ConicSettings | None):
    resolved = ConicSettings() if settings is None else settings
    if int(gamma) == 1:
        return Gamma1Model(quotes, float(K), support_cap=support_cap, settings=resolved)
    if int(gamma) == 2:
        return Gamma2Model(quotes, float(K), support_cap=support_cap, settings=resolved)
    raise ValueError(f"unsupported gamma {gamma}; expected 1 or 2")


def _resolve_support_cap(gamma: int, support_cap) -> float | None:
    if support_cap is None:
        return None
    if isinstance(support_cap, str):
        text = support_cap.strip().lower()
        if text in {"auto"}:
            return DEFAULT_SUPPORT_CAP if int(gamma) > 0 else None
        if text in {"none", "null", "unbounded"}:
            return None
        value = float(text)
        if not np.isfinite(value):
            return None
        return value
    value = float(support_cap)
    if not np.isfinite(value):
        return None
    return value


def _infeasible_error(
    cs: CrossSection,
    quote_set: str,
    gamma: int,
    support_cap: float | None,
    status: str,
    *,
    detail: str = "",
) -> InfeasibleCrossSectionError:
    date = pd.Timestamp(cs.date)
    expiry = pd.Timestamp(cs.expiry)
    cap_text = "none" if support_cap is None else f"{float(support_cap):g}"
    extra = ""
    if detail and str(detail) != str(status):
        extra = f": {detail}"
    hint = _INFEASIBLE_HINT if int(gamma) > 0 else ""
    message = (
        f"infeasible cross-section {_iso_date(date)} → {_iso_date(expiry)} "
        f"(quote_set={quote_set}, gamma={int(gamma)}, support_cap={cap_text}, "
        f"status={status}){extra}{hint}"
    )
    return InfeasibleCrossSectionError(
        message,
        date=date,
        expiry=expiry,
        quote_set=str(quote_set),
        gamma=int(gamma),
        support_cap=None if support_cap is None else float(support_cap),
        status=str(status),
    )


def _degenerate_physical(model, settings: LPSettings):
    points: list[np.ndarray] = []
    planes: list[tuple[np.ndarray, float]] = []
    violations: list[float] = []
    messages: list[str] = []
    for direction in _AXIS_DIRECTIONS:
        result = model.support(direction)
        if not result.ok:
            raise SolverFailure(
                f"support failed for direction {direction[0]}, {direction[1]}: {result.status}"
            )
        points.append(np.asarray(result.point, dtype=float).reshape(2).copy())
        planes.append((direction.copy(), float(result.value)))
        violations.append(float(result.violation))
        messages.append(str(result.status))
    tolerance = max(float(settings.point_tolerance), float(settings.degenerate_tolerance))
    inner, dimension, status = _classify_axis_cloud(points, tolerance)
    return inner, inner.copy(), _raw_halfplanes(planes), dimension, status, 4, violations, messages


def _resolve_quotes(
    cs: CrossSection,
    K,
    quote_set: str | pd.DataFrame | None,
    design: str | None,
    gamma: int,
) -> tuple[pd.DataFrame, str, str]:
    resolved_design = design
    if resolved_design is None:
        resolved_design = "riskneutral" if int(gamma) == 0 else "physical"
    if isinstance(quote_set, pd.DataFrame):
        return quote_set.copy().reset_index(drop=True), _CUSTOM_SET_NAME, resolved_design
    name = "put_wing" if quote_set is None and int(gamma) == 0 else (
        "full_otm" if quote_set is None else str(quote_set)
    )
    return select_quote_set(cs.quotes, K, name, resolved_design), name, resolved_design


def _configure_gamma0(
    quotes: pd.DataFrame,
    K,
    *,
    mode: str,
    budget: float | None,
    support_cap: float | None,
    lp_settings: LPSettings,
    full_strike_count: int | None,
    warn_relaxed: bool,
) -> tuple[MeasureProblem, ConstraintSystem, bool]:
    try:
        problem = build_problem(quotes, float(K), full_strike_count, support_cap=support_cap)
    except ValueError as exc:
        text = str(exc)
        if support_cap is not None and (
            "positive call bid is incompatible with support cap" in text
            or "support cap leaves fewer than two strikes" in text
        ):
            raise InfeasibleCrossSectionError(
                f"support_cap={float(support_cap):g} makes the risk-neutral system infeasible: {text}"
            ) from exc
        raise
    fallback = False
    if mode == "auto":
        system = configure_constraints(problem, lp_settings, "hard")
        if system is None:
            system = configure_constraints(problem, lp_settings, "relaxed", budget)
            fallback = True
            if system is not None and warn_relaxed:
                logger.warning(_RELAXED_WARNING)
    elif mode in {"hard", "relaxed"}:
        system = configure_constraints(problem, lp_settings, mode, budget)
    else:
        raise ValueError(f"unrecognized constraint mode {mode!r}")
    if system is None:
        raise InfeasibleCrossSectionError(
            f"quotes admit no feasible risk-neutral measure (mode={mode}, n_quotes={len(problem.quotes)})"
        )
    if support_cap is not None:
        system = impose_support_cap(system, float(support_cap))
        constrained = system.problem
        feasible = solve_lp(
            np.zeros(constrained.slack_objective.shape, dtype=float),
            system.A_ub,
            system.b_ub,
            constrained.A_eq,
            constrained.b_eq,
            constrained.bounds,
            lp_settings,
        )
        if not feasible.success:
            if int(feasible.status) == 2:
                raise InfeasibleCrossSectionError(
                    f"support_cap={float(support_cap):g} makes the risk-neutral system infeasible"
                )
            raise SolverFailure(f"support-cap feasibility check failed: {feasible.message}")
    return problem, system, fallback


def _interior_state(
    problem: MeasureProblem, system: ConstraintSystem, settings: LPSettings
) -> tuple[bool, float, str]:
    if system.mode != "hard":
        return False, float("nan"), "relaxed"
    success, eta_star, message = interior_margin(problem, settings)
    if not success:
        return False, float("nan"), message
    widths = (
        problem.quotes["norm_u"].to_numpy(dtype=float)
        - problem.quotes["norm_l"].to_numpy(dtype=float)
    )
    ok = bool(
        np.all(np.isfinite(widths))
        and np.all(widths > 0.0)
        and np.isfinite(eta_star)
        and float(eta_star) > settings.interior_cutoff
    )
    return ok, float(eta_star), message


def _degenerate_polygon(system: ConstraintSystem, settings: LPSettings):
    points: list[np.ndarray] = []
    planes: list[tuple[np.ndarray, float]] = []
    gaps: list[float] = []
    messages: list[str] = []
    for direction in _AXIS_DIRECTIONS:
        result = support(system, settings, direction, lexicographic=True)
        if not result.success:
            raise SolverFailure(
                f"support failed for direction {direction[0]}, {direction[1]}: {result.message}"
            )
        points.append(np.asarray(result.point, dtype=float).reshape(2).copy())
        planes.append((direction.copy(), float(result.upper_value)))
        gaps.append(float(result.duality_gap))
        messages.append(str(result.message))
    tolerance = max(float(settings.point_tolerance), float(settings.degenerate_tolerance))
    inner, dimension, status = _classify_axis_cloud(points, tolerance)
    return inner, inner.copy(), _raw_halfplanes(planes), dimension, status, 4, gaps, messages


def _classify_axis_cloud(points, tolerance: float) -> tuple[np.ndarray, int, str]:
    """Polygon, affine dimension, and status from axis-aligned support points.

    A two-dimensional cloud stays a polygon. It is not collapsed to a segment.
    """
    unique = unique_points(points, tolerance)
    dimension = affine_dimension(unique, tolerance)
    if dimension <= 0:
        polygon = unique[:1] if len(unique) else np.zeros((1, 2), dtype=float)
        return _as_cloud(polygon), 0, "point"
    if dimension == 1:
        return _as_cloud(segment_endpoints(unique)), 1, "segment"
    return _as_cloud(convex_hull(unique)), 2, "polygon"


def _box_normalizer(
    lo: np.ndarray,
    hi: np.ndarray,
    degenerate_tolerance: float,
    normalizer: BoxNormalizer | None,
) -> BoxNormalizer:
    if normalizer is not None:
        return normalizer
    scale = np.asarray(hi - lo, dtype=float).reshape(2)
    scale = np.where(scale <= float(degenerate_tolerance), 1.0, scale)
    return BoxNormalizer(lo, scale)


def _area_predicate(settings: LPSettings, normalizer: BoxNormalizer, p0: float):
    absolute = float(settings.anytime_area_absolute_tolerance)
    relative = float(settings.anytime_area_relative_tolerance)

    def predicate(inner: np.ndarray, outer: np.ndarray) -> bool:
        inner_area, _, _ = transformed_area(normalizer.to_raw(inner), float(p0))
        outer_area, _, _ = transformed_area(normalizer.to_raw(outer), float(p0))
        if not np.isfinite(inner_area) or not np.isfinite(outer_area) or outer_area < inner_area:
            return False
        return (outer_area - inner_area) <= absolute + relative * outer_area

    return predicate


def _areas(inner: np.ndarray, outer: np.ndarray, p0: float) -> tuple[float, float]:
    area, _, _ = transformed_area(inner, float(p0))
    outer_area, _, _ = transformed_area(outer, float(p0))
    return float(area), float(outer_area)


def _reported_areas(
    inner: np.ndarray,
    outer: np.ndarray,
    p0: float,
    baseline: float,
    dimension: int,
) -> tuple[float, float, float]:
    """Inner area, true outer area, and area ratio.

    Dimension below 2 has no transformed area ratio. The outer area is the
    value returned by ``transformed_area``, including ``nan`` and ``inf``.
    """
    if int(dimension) < 2:
        return 0.0, 0.0, float("nan")
    area, outer_area = _areas(inner, outer, p0)
    return float(area), float(outer_area), float(area_ratio(area, baseline))


def _depth_range(
    points: np.ndarray,
    *,
    K: float | None = None,
    probability_floor: float = 0.0,
) -> tuple[float, float]:
    cloud = _as_cloud(points)
    if len(cloud) == 0:
        return float("nan"), float("nan")
    positive = cloud[:, 0] > max(0.0, float(probability_floor))
    if not np.any(positive):
        return float("nan"), float("nan")
    probabilities = cloud[positive, 0]
    losses = cloud[positive, 1]
    if K is not None:
        losses = np.clip(losses, 0.0, float(K) * probabilities)
    depths = losses / probabilities
    return float(np.min(depths)), float(np.max(depths))


def _clip_to_target_wedge(points: np.ndarray, K: float) -> np.ndarray:
    polygon = _as_cloud(points)
    if len(polygon) == 0:
        return polygon
    clipped = polygon
    for normal, offset in (
        (np.array([-1.0, 0.0]), 0.0),
        (np.array([1.0, 0.0]), 1.0),
        (np.array([0.0, -1.0]), 0.0),
        (np.array([-float(K), 1.0]), 0.0),
    ):
        clipped = clip_polygon(clipped, normal, offset)
        if len(clipped) == 0:
            return clipped
    return _as_cloud(clipped)


def _validate_depth_enclosure(
    exact_lo: float,
    exact_hi: float,
    inner_lo: float,
    inner_hi: float,
    *,
    gamma: int,
    quote_set: str,
) -> bool:
    del gamma, quote_set
    values = np.array([exact_lo, exact_hi, inner_lo, inner_hi], dtype=float)
    if not np.all(np.isfinite(values)):
        return True
    tolerance = 5e-3
    return not (
        inner_lo < exact_lo - tolerance or inner_hi > exact_hi + tolerance
    )


def _nested_names(sets, gamma: int) -> tuple[str, ...]:
    if sets is None:
        if int(gamma) == 0:
            return ("put_pair", "local_eight", "put_wing")
        return ("put_pair", "local_eight", "put_wing", "full_otm")
    names = tuple(str(name) for name in sets)
    if not names:
        raise ValueError("sets must contain at least one quote-set name")
    return names


def _apply_accumulated_halfplanes(
    frontier: Frontier, accumulated: list[tuple[np.ndarray, float]]
) -> Frontier:
    planes = list(accumulated)
    planes.extend(_halfplane_pairs(frontier.halfplanes))
    if not planes:
        return frontier
    normals = np.vstack([np.asarray(normal, dtype=float).reshape(2) for normal, _offset in planes])
    offsets = np.asarray([float(offset) for _normal, offset in planes], dtype=float)
    box_normals = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=float,
    )
    box_offsets = np.array(
        [float(frontier.hi[0]), -float(frontier.lo[0]), float(frontier.hi[1]), -float(frontier.lo[1])],
        dtype=float,
    )
    try:
        outer = halfplane_intersection(
            np.vstack([normals, box_normals]),
            np.concatenate([offsets, box_offsets]),
            interior_points=frontier.inner,
        )
    except ValueError:
        outer = _as_cloud(frontier.outer)
        for normal, offset in planes:
            outer = clip_polygon(outer, normal, offset)
            if len(outer) == 0:
                break
    frontier.halfplanes = _raw_halfplanes(planes)
    clipped_inner = _as_cloud(frontier.inner)
    for normal, offset in planes:
        clipped_inner = clip_polygon(clipped_inner, normal, offset)
        if len(clipped_inner) == 0:
            break
    frontier.inner = _clip_to_target_wedge(clipped_inner, frontier.K)
    frontier.outer = _clip_to_target_wedge(_as_cloud(outer), frontier.K)
    if int(frontier.dimension) >= 2:
        area, area_outer = _areas(frontier.inner, frontier.outer, frontier.p0)
        frontier.area = float(area)
        frontier.area_outer = float(area_outer)
        frontier.area_ratio = float(area_ratio(area, frontier.baseline_area))
    return frontier


def _halfplane_pairs(halfplanes: np.ndarray) -> list[tuple[np.ndarray, float]]:
    planes = _as_halfplanes(halfplanes)
    return [(row[:2].copy(), float(row[2])) for row in planes]


def _unit_halfplanes_to_raw(
    normalizer: BoxNormalizer, certificates: list[tuple[np.ndarray, float]]
) -> np.ndarray:
    rows = []
    for direction, bound in certificates:
        raw_direction, raw_offset = normalizer.raw_halfplane(direction, bound)
        rows.append([float(raw_direction[0]), float(raw_direction[1]), float(raw_offset)])
    return _as_halfplanes(rows)


def _grid_halfplanes_to_raw(normalizer: BoxNormalizer, supports: pd.DataFrame) -> np.ndarray:
    rows = []
    for row in supports.itertuples(index=False):
        direction = np.array([float(row.direction_1), float(row.direction_2)], dtype=float)
        raw_direction, raw_offset = normalizer.raw_halfplane(direction, float(row.support))
        rows.append([float(raw_direction[0]), float(raw_direction[1]), float(raw_offset)])
    return _as_halfplanes(rows)


def _raw_halfplanes(certificates: list[tuple[np.ndarray, float]]) -> np.ndarray:
    rows = [
        [float(direction[0]), float(direction[1]), float(bound)]
        for direction, bound in certificates
    ]
    return _as_halfplanes(rows)


def _as_cloud(points) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.atleast_2d(values).reshape(-1, 2).copy()


def _as_halfplanes(values) -> np.ndarray:
    planes = np.asarray(values, dtype=float)
    if planes.size == 0:
        return np.empty((0, 3), dtype=float)
    return np.atleast_2d(planes).reshape(-1, 3).copy()


def _iso_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _jsonify(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return _iso_date(value)
    if isinstance(value, pd.DataFrame):
        return _jsonify(value.to_dict(orient="records"))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _SupportOracle:
    def __init__(
        self, system: ConstraintSystem, settings: LPSettings, normalizer: BoxNormalizer
    ) -> None:
        self.system = system
        self.settings = settings
        self.normalizer = normalizer
        self.duality_gaps: list[float] = []
        self.statuses: list[str] = []

    def __call__(self, unit_direction) -> tuple[np.ndarray, float]:
        direction = np.asarray(unit_direction, dtype=float).reshape(2)
        raw = self.normalizer.raw_direction(direction)
        result = support(self.system, self.settings, raw, lexicographic=True)
        if not result.success:
            raise SolverFailure(
                f"support failed for direction {raw[0]}, {raw[1]}: {result.message}"
            )
        self.duality_gaps.append(float(result.duality_gap))
        self.statuses.append(str(result.message))
        return (
            self.normalizer.to_unit(result.point),
            self.normalizer.unit_support(float(result.upper_value), direction),
        )


class _PhysicalOracle:
    def __init__(self, model, normalizer: BoxNormalizer) -> None:
        self.model = model
        self.normalizer = normalizer
        self.violations: list[float] = []
        self.statuses: list[str] = []

    def __call__(self, unit_direction) -> tuple[np.ndarray, float]:
        direction = np.asarray(unit_direction, dtype=float).reshape(2)
        raw = self.normalizer.raw_direction(direction)
        result = self.model.support(raw)
        if not result.ok:
            raise SolverFailure(
                f"support failed for direction {raw[0]}, {raw[1]}: {result.status}"
            )
        self.violations.append(float(result.violation))
        self.statuses.append(str(result.status))
        return (
            self.normalizer.to_unit(result.point),
            self.normalizer.unit_support(float(result.value), direction),
        )


def _n_strikes(quotes: pd.DataFrame) -> int:
    if quotes is None or len(quotes) == 0:
        return 0
    if "strike" in quotes.columns:
        return int(quotes["strike"].nunique())
    if "x" in quotes.columns:
        return int(quotes["x"].nunique())
    return 0
