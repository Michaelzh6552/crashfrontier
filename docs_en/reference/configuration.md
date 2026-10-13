# Configuration

Solver dataclasses and the packaged defaults that public functions use when
you omit a keyword. Import settings from the package root; import the
`DEFAULT_*` constants from `crashfrontier.config` (they are not in
`crashfrontier.__all__`).

```python
from crashfrontier import LPSettings, GridSettings, ConicSettings, compute_frontier

frontier = compute_frontier(
    cs,
    K=0.90,
    lp_settings=LPSettings(time_limit_seconds=10.0),
    grid_settings=GridSettings(levels=(32, 64)),
    conic_settings=ConicSettings(solver="CLARABEL"),
)
```

Omitting `lp_settings`, `grid_settings`, or `conic_settings` is the same as
passing the zero-argument constructor. All three dataclasses are frozen.

Which object is read depends on `gamma` and `method`. Unused objects are
still instantiated and then ignored.

| Object | `gamma=0` hull | `gamma=0` grid | `gamma>0` grid | `gamma>0` hull | Degenerate box |
|---|---|---|---|---|---|
| `LPSettings` | yes | tolerances + validation | degenerate test only | hull + degenerate | yes |
| `GridSettings` | no | yes | yes | no | no |
| `ConicSettings` | no | no | yes | yes | physical supports |

`mode` and `budget` apply only at `gamma=0`. `p0` affects transformed area
and `depth_boundary`; it does not change either the joint set or the reported
depth endpoints.

## LPSettings

```python
LPSettings(
    primal_feasibility_tolerance: float = 1e-10,
    dual_feasibility_tolerance: float = 1e-10,
    time_limit_seconds: float = 5.0,
    edge_tolerance: float = 1e-8,
    point_tolerance: float = 1e-9,
    degenerate_tolerance: float = 1e-10,
    max_support_calls: int = 500,
    anytime_max_support_calls: int = 160,
    anytime_area_relative_tolerance: float = 0.002,
    anytime_area_absolute_tolerance: float = 1e-12,
    interior_cutoff: float = 1e-8,
    validation_multiplier: float = 5.0,
)
```

Risk-neutral LPs use SciPy `linprog` with HiGHS **dual simplex**
(`highs-ds`), then HiGHS **interior point** (`highs-ipm`) if the first
status is `4` or the message contains `model_status is Unknown`.

| Field | When it matters |
|---|---|
| `primal_feasibility_tolerance`, `dual_feasibility_tolerance` | Passed to HiGHS |
| `time_limit_seconds` | HiGHS `time_limit` |
| `validation_multiplier` | A recovered LP is accepted only if equality/inequality residuals stay within `max(5e-10, validation_multiplier * primal_feasibility_tolerance)` and `‖x‖_∞ ≤ 1e6` |
| `interior_cutoff` | `gamma=0` and `mode="hard"`: `interior_ok` requires every quote width `> 0` and `η* > interior_cutoff`. Relaxed systems set `interior_ok=False` and `interior_margin=nan` |
| `degenerate_tolerance` | If any marginal width `prob_hi-prob_lo` or `loss_hi-loss_lo` is `≤` this value, reconstruction is skipped (`method="degenerate"`, area `0`) |
| `edge_tolerance`, `point_tolerance` | Hull edge certification and point uniqueness |
| `max_support_calls` | Hull budget for hard `gamma=0` and all `gamma>0` hulls |
| `anytime_max_support_calls`, `anytime_area_*` | **Only** `gamma=0` and `mode="relaxed"`: hull stops early when `area(outer)-area(inner) ≤ absolute + relative * area(outer)` (`status="area_certified"`) |

Physical models still read `degenerate_tolerance` and, if `method="hull"`,
the hull fields. They do not use the anytime-area predicate.

## GridSettings

```python
GridSettings(
    levels: tuple[int, ...] = (32, 64, 128),
    max_normalized_gap: float = 0.01,
)
```

Used when `method="grid"`, or when `method="auto"` and `gamma>0` (the
physical default).

| Field | When it matters |
|---|---|
| `levels` | Equispaced direction counts tried in order. Empty `levels` raises `ValueError` |
| `max_normalized_gap` | Stop at the first level whose unit-box support gap is `≤` this value (`status="converged"`). Otherwise the last level is kept (`status="max_level"`) |

The gap is measured in the **normalized** box (`BoxNormalizer`), not in raw
`(prob, loss)`. Raising `max_normalized_gap` or shortening `levels` is the
usual way to cheapen a physical solve.

## ConicSettings

```python
ConicSettings(
    solver: str = "CLARABEL",
    fallbacks: tuple[str, ...] = ("ECOS", "SCS"),
    tolerance: float = 1e-9,
    max_iterations: int = 2000,
    violation_tolerance: float = 1e-6,
    objective_tolerance: float = 1e-7,
    warm_start: bool = True,
)
```

Used only at `gamma>0`. Import is `from crashfrontier import ConicSettings`
(same object as `crashfrontier.config.ConicSettings` /
`crashfrontier.physical.solve.ConicSettings`).

| Field | When it matters |
|---|---|
| `solver` | First CVXPY solver. Names are upper-cased |
| `fallbacks` | Tried in order if installed and different from `solver`. ECOS and SCS require `crashfrontier[solvers]` |
| `tolerance`, `max_iterations` | Mapped for **CLARABEL** (`tol_gap_abs/rel`, `tol_feas`, `max_iter`) and **ECOS** (`abstol`, `reltol`, `feastol`, `max_iters`) |
| `violation_tolerance` | `ok` requires a finite value and cone residual `≤` this |
| `objective_tolerance` | Two `OPTIMAL_INACCURATE` results are accepted only when their objectives agree within `objective_tolerance * (1 + |v|)` |
| `warm_start` | First attempt may warm-start; inaccurate results get a **cold** restart on the primary solver before fallbacks |

**SCS** does not read `tolerance` or `max_iterations`. It always uses
`eps=1e-6` and `max_iters=20000`.

`OPTIMAL` with a finite value and a small residual is accepted immediately.
Exact `INFEASIBLE` / `UNBOUNDED` statuses are terminal.
`OPTIMAL_INACCURATE` and `INFEASIBLE_INACCURATE` trigger a cold restart and
installed fallbacks. An inaccurate infeasibility needs confirmation from a
different solver; disagreeing inaccurate optima remain uncertified. Other
failures, including solver errors, also go to fallbacks. `ok` is false when no
certified attempt remains.

Physical `interior_ok` uses cutoff `1e-10` (not `LPSettings.interior_cutoff`).
A failed interior test logs a warning and **does not** abort the frontier.

## Function defaults {#defaults}

These constants live on `crashfrontier.config`. Public functions in-line the
same numbers, so you do not need to import them.

| Constant | Value | Used as |
|---|---|---|
| `DEFAULT_THRESHOLD` | `0.90` | `K` on compute / batch / CLI |
| `DEFAULT_P0` | `1e-4` | Transformed-area / depth-boundary floor |
| `DEFAULT_SUPPORT_CAP` | `5.0` | `support_cap="auto"` when `gamma>0` |
| `DEFAULT_LM_RANGE` | `(-0.35, 0.05)` | `Dataset.expiries` / `cross_section` log-moneyness window; put-universe for risk-neutral sets |
| `DEFAULT_MEMORY_LIMIT` | `512 * 1024 * 1024` | `Dataset(memory_limit=...)`; CLI treats larger **text** option files as needing `crashfrontier index` |
| `DEFAULT_TENORS` | `(7, 14, 30, 60, 90)` | `term_structure` |
| `DEFAULT_WINDOW_FRACTION` | `0.35` | `term_structure` window `round(tenor * (1 ± fraction))`, clipped to the filter day limits |
| `DEFAULT_SERIES_TARGET` | `30` | `frontier_series(target_days=...)` |
| `DEFAULT_SERIES_WINDOW` | `(20, 45)` | `frontier_series(window=...)` |
| `SERIES_CACHE_FLUSH` | `20` | Write `cache_path` every 20 newly computed rows |
| `WEEKDAY_PRIORITY` | `(2, 1, 3, 0, 4)` | Weekly sampling: Wednesday, Tuesday, Thursday, Monday, Friday |

`support_cap` strings accepted by `compute_frontier`: `"auto"`, `"none"`,
`"null"`, `"unbounded"`. `"auto"` is `5.0` at `gamma>0` and unbounded at
`gamma=0`. Non-finite numbers become unbounded.

Display maps `SET_ORDER`, `COLORS`, `LABELS`, and `PALETTE` are used by
plotting, not by the solver.

## Data-side knobs

Not solver settings, but they change the program:

| Knob | Default | Effect |
|---|---|---|
| `QuoteFilter` | `min_volume=1`, days `[7, 120]`, uncrossed book | Rows that never reach `compute_frontier` |
| `rate_unit` | `"percent"` | File/frame rates are divided by 100. A **numeric** `rates=0.02` is already decimal and ignores `rate_unit` |
| `forward_sources` | `("table", "column", "parity")` | First finite forward wins |
| `strike_scale` | `"auto"` on `Dataset`; `1.0` on `CrossSection.from_quotes` | `auto` chooses among `1, 10, 100, 1000` so median strike matches a reference price |

See [Python API](api.md#data) and the [data guide](../data-guide/index.md).

Related pages: [Python API](api.md), [results](results.md), [exceptions](errors.md).
