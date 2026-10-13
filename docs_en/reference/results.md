# Results and schemas

Return objects, table columns, units, status tokens, and serialization.
Constructors for `compute_frontier` and friends are in the [Python API](api.md).
Exceptions are in [exceptions](errors.md).

```python
from crashfrontier import compute_frontier
from crashfrontier.riskneutral import DualCertificate  # not a package-root name

frontier = compute_frontier(cs, K=0.90)
print(frontier.describe())
print(frontier.summary()["prob_lo"], frontier.summary()["loss_hi"])
print(frontier.contains(0.02, 0.004))
```

## Units {#units}

Coordinates are **normalized to the forward** `F`. `K` and strike moneyness
`x` are `strike / F`. Loss is `E[(K - X)^+]` in units of `F`.

| Quantity | Stored in `Frontier`, CSV, JSON, `scenario_check` | Figures |
|---|---|---|
| Probability (`prob`, `p`) | Decimal, typically in `[0, 1]` | Percent (`×100`) |
| Expected loss (`loss`, `L`) | Decimal relative to `F` | Basis points by default (`×10000`); `loss_unit="percent"` is `×100` |
| Conditional depth (`depth`, `s = L/p`) | Decimal, typically in `[0, K]` | Percent (`×100`) |
| Area / `area_ratio` | Transformed `(p, L/p)` area for `p ≥ p0` | Not plotted as a number |
| `rate` | Decimal continuously compounded | — |
| `discount` | `exp(-rate * days / 365)` | — |
| `support_cap` | Upper bound on normalized spot `X`; `None` is unbounded | — |

`p0` (default `1e-4`) clips the transformed area and `depth_boundary`. It
does not change `inner` / `outer` vertices.

At `gamma>0`, `depth_lo` / `depth_hi` come from a separate SOCP. They can
be NaN (for example when `prob_hi < 1e-4`) while the joint set is still
populated. Vertex-table `depth = loss / prob` is then geometry, not a
certified interval.

"Certified" throughout this reference means certified up to the solver
tolerances configured in `LPSettings` / `ConicSettings` (feasibility,
duality gap, cone violation); no numerical bound is exact at arbitrary
precision.

## Frontier {#frontier}

`crashfrontier.Frontier` is a mutable dataclass. Users normally receive it
from `compute_frontier` / `nested_frontiers`; there is no public builder.

### Identity and market

| Field | Type | Meaning |
|---|---|---|
| `date`, `expiry` | `pd.Timestamp` | Observation date and expiry (normalized) |
| `days` | `int` | `expiry - date` in calendar days |
| `am_settlement` | `int` | `0` PM, `1` AM |
| `K` | `float` | Crash threshold in `x` units |
| `gamma` | `int` | `0` risk-neutral, `1` / `2` physical |
| `support_cap` | `float` or `None` | Resolved cap |
| `quote_set` | `str` | Named set, or `"custom"` if a `DataFrame` was passed |
| `n_quotes`, `n_strikes` | `int` | Rows and distinct strikes **that entered the program** |
| `forward`, `spot` | `float` | `spot` may be NaN |
| `forward_source` | `str` | `table`, `column`, `parity`, `user` (or `none` only on `expiries` tables) |
| `rate`, `discount` | `float` | See [units](#units) |

### Reconstruction

| Field | Type | Meaning |
|---|---|---|
| `mode` | `str` | `hard` / `relaxed` (`gamma=0`) or `socp` (`gamma>0`) |
| `method` | `str` | `hull`, `grid`, or `degenerate` |
| `status` | `str` | Certification token; see [statuses](#statuses) |
| `dimension` | `int` | Affine dimension of the recovered set (`0`, `1`, or `2`) |
| `inner`, `outer` | `ndarray`, shape `(n, 2)` | Vertex clouds `(prob, loss)` |
| `halfplanes` | `ndarray`, shape `(m, 3)` | Rows `(a_prob, a_loss, h)` meaning `a_prob * p + a_loss * L <= h` in raw coordinates |
| `lo`, `hi` | `ndarray`, shape `(2,)` | Marginal box `[prob, loss]` |
| `depth_lo`, `depth_hi` | `float` | Certified or geometric depth interval (see above) |
| `p0` | `float` | Transform floor |
| `area`, `area_outer` | `float` | Transformed areas; `0` when `method="degenerate"`; NaN if the `p0` clip has fewer than three vertices |
| `baseline_area` | `float` | Transformed area of the marginal box cut by `0 ≤ L ≤ K p` and `p ≥ p0` |
| `area_ratio` | `float` | `area / baseline_area`, or NaN for a degenerate set or non-positive baseline |
| `interior_ok` | `bool` | Uniform bid–ask shrink remains feasible |
| `interior_margin` | `float` | `η*` on `[0, 1/2]`; NaN when unavailable |
| `diagnostics` | `dict` | See [diagnostics](#diagnostics) |
| `normalizer` | `BoxNormalizer` | Affine map used during reconstruction (`origin`, `scale`) |
| `quotes` | `pd.DataFrame` | Program quotes; columns in [quote table](#quotes) |

Both `inner` and `outer` are clipped to the target wedge
`0 ≤ p ≤ 1`, `L ≥ 0`, `L ≤ K p`.
`nested_frontiers(..., accumulate_halfplanes=True)` intersects both polygons
with accumulated half-planes, replaces the stored half-plane set, and
recomputes `area`, `area_outer`, and `area_ratio`.

### Methods

```python
frontier.describe() -> str
frontier.summary() -> dict[str, object]
frontier.vertices() -> pd.DataFrame
frontier.halfplane_table() -> pd.DataFrame
frontier.contains(prob, loss) -> str
frontier.depth_boundary(samples_per_edge=80, *, p_floor=None) -> np.ndarray
frontier.to_dict() -> dict
```

| Method | Returns |
|---|---|
| `describe()` | Six-line text (dates, `K`, `γ`, quotes, `F`, `D`, mode/method/status, intervals, areas) |
| `summary()` | Scalar record in the [fixed column order](#summary) |
| `vertices()` | Inner then outer vertices |
| `halfplane_table()` | Empty frame with the three columns if `halfplanes` is empty |
| `contains(prob, loss)` | `inside` / `outside` / `boundary` |
| `depth_boundary(...)` | Closed `(p, L/p)` polyline for `p ≥ p_floor` (`p0` when `p_floor` is omitted) |
| `to_dict()` | JSON-ready record; see [serialization](#todict) |

`contains`:

- `inside` — point is in the **inner** polygon, including its boundary
- `outside` — point is strictly outside the **outer** polygon
- `boundary` — the remaining shell, including the outer boundary when it
  differs from the inner set

## Statuses {#statuses}

### `mode`

| Value | When |
|---|---|
| `hard` | `gamma=0` zero-slack system |
| `relaxed` | `gamma=0` slack-budget system (`δ*` or caller `budget`) |
| `socp` | `gamma>0` |

### `method`

| Value | When |
|---|---|
| `hull` | `method="hull"`, or `method="auto"` at `gamma=0` |
| `grid` | `method="grid"`, or `method="auto"` at `gamma>0` |
| `degenerate` | A marginal width `≤ LPSettings.degenerate_tolerance` (`1e-10` by default) |

### `status` (reconstruction)

Written to `Frontier.status` and `summary()["status"]`.

| Value | Reconstruction | Meaning |
|---|---|---|
| `edges_certified` | hull | Every inner edge passed `edge_tolerance` |
| `area_certified` | hull | Anytime area predicate (relaxed `gamma=0` only) |
| `budget_reached` | hull | `max_support_calls` / `anytime_max_support_calls` exhausted |
| `stalled` | hull | New support points did not expand the unique set |
| `point` | hull or degenerate | Affine dimension `0` |
| `segment` | hull or degenerate | Affine dimension `1` |
| `converged` | grid | Unit-box gap `≤ max_normalized_gap` |
| `max_level` | grid | Last direction count still exceeds the gap tolerance |

`dimension` is `2` for certified/grid polygons, `0`/`1` for point/segment.
Degenerate `area` / `area_outer` are `0`; `area_ratio` is NaN because a
point or segment has no two-dimensional identification-area ratio.

`diagnostics["statuses"]` is a list that also records constraint mode,
interior-test tokens, and per-support solver messages. Do not treat that
list as the scalar `status`.

## Diagnostics {#diagnostics}

Always present:

| Key | `gamma=0` | `gamma>0` |
|---|---|---|
| `support_calls` | `int` | `int` (`4` if degenerate) |
| `max_gap` | hull/grid gap; `0` if degenerate | same |
| `elapsed_seconds` | wall time | wall time |
| `statuses` | `list[str]` | `list[str]` |
| `relaxed` | `True` if `mode="auto"` fell back | always `False` |
| `budget` | slack budget or `None` | `None` |
| `minimum_slack` | `δ*` | `nan` |
| `max_duality_gap` | max LP dual gap | `nan` |
| `max_violation` | absent | max cone residual |

`summary()["solver_gap"]` is `max_duality_gap` at `gamma=0` and
`max_violation` at `gamma>0`. Missing values become NaN.

Physical-only keys: `solver_depth_lo`, `solver_depth_hi`,
`inner_depth_lo`, `inner_depth_hi`, `depth_enclosure_ok`,
`depth_reconciled`, and
`depth_endpoint_gap` (the maximum absolute gap between the raw SOCP endpoints
and inner-vertex depths). If the inner range extends outside the raw SOCP
range by at most `5e-3`, the public interval expands to enclose it. A larger
outward extension sets `depth_enclosure_ok=False` and the public endpoints to
NaN. A wider SOCP range that contains the inner range remains valid even when
`depth_endpoint_gap` exceeds `5e-3`.
`inner_depth_probability_floor` records the solver-residual-based cutoff used
to avoid dividing numerical noise by a near-zero probability.

## Tables {#tables}

### Summary {#summary}

`Frontier.summary()` key order is fixed and matches
`term_structure` / `threshold_profile` / `frontier_series` leading columns:

```text
date, expiry, days, am_settlement, K, gamma, support_cap, quote_set,
n_quotes, n_strikes, forward, forward_source, rate, discount, spot,
mode, method, status, dimension, prob_lo, prob_hi, loss_lo, loss_hi,
depth_lo, depth_hi, area, area_outer, baseline_area, area_ratio,
interior_ok, interior_margin, support_calls, max_gap, solver_gap,
elapsed_seconds
```

`write_frontier` / `frontier_frames` insert `name` at the front. Batch
helpers append:

| Function | Extra columns |
|---|---|
| `term_structure` | `target_days`, `error` |
| `threshold_profile` | `error` |
| `frontier_series` | `requested_date`, `error` |

A failed batch row sets most numerics to NaN, `date`/`expiry` to `NaT`
unless a `CrossSection` was already built, string fields to NaN, and
`error` to the exception message. Successful rows have `error=""` (empty
string, not NaN). Filter with `table[table["error"].fillna("").eq("")]`.

`term_structure` writes `error="expiry already used"` when two tenors
collapse to the same expiry; the tenor closer to its target is kept.

### Vertices

| Column | Type | Meaning |
|---|---|---|
| `kind` | `str` | `inner` or `outer` |
| `order` | `int` | Index along that cloud |
| `prob`, `loss` | `float` | Raw decimals |
| `depth` | `float` | `loss / prob` if `prob > 0`, else NaN |

Export adds `name`.

### Half-planes

| Column | Meaning |
|---|---|
| `a_prob`, `a_loss`, `h` | `a_prob * p + a_loss * L <= h` |

Export adds `name`.

### Quotes {#quotes}

`Frontier.quotes` is the filtered set that entered the LP/SOCP, not the
full `CrossSection.quotes` unless those coincide. Columns:

```text
quote_id, symbol, root, cp_flag, strike, bid, ask, volume, open_interest,
impl_volatility, x, log_moneyness, norm_bid, norm_ask, norm_l, norm_u, role
```

`x = strike / F`. `norm_bid` / `norm_ask` are `bid` / `ask` divided by
`discount * F`. Puts are mapped to call quotes:
`norm_l = norm_bid + 1 - x` (same for the ask). `role` is `otm_put`
(`P` and `x <= 1`), `otm_call` (`C` and `x >= 1`), or `itm`.

Export adds `name`.

### CrossSection.summary()

Not a frontier table. Keys: `date`, `expiry`, `am_settlement`, `root`,
`days`, `tau`, `forward`, `forward_source`, `rate`, `rate_source`,
`discount`, `spot`, `n_quotes`, `n_puts`, `n_calls`, `n_put_strikes`,
`n_call_strikes`, `ambiguous_settlement`.

`Dataset.expiries` columns: `expiry`, `am_settlement`, `root`, `days`,
`n_rows`, `n_put_strikes`, `n_call_strikes`, `forward`, `forward_source`.

`Dataset.describe()` keys: `options` (`files`, `kind`, `n_dates`,
`first_date`, `last_date`, `columns`), `date_format`, `strike_scale`,
`forwards`, `rates`, `spot`, `quote_filter`, `cache_dir`.

## DualCertificate {#certificate}

Returned by `binding_quotes`. Import:

```python
from crashfrontier.riskneutral import DualCertificate
```

Defined only for `gamma=0`. The hard system is used when feasible;
otherwise the slack-budget system is used and a warning is logged.

| Field | Type | Meaning |
|---|---|---|
| `direction` | `ndarray` shape `(2,)` | Raw `(u_p, u_L)` |
| `support_value`, `upper_value` | `float` | Primal support and duality-gap envelope |
| `duality_gap` | `float` | Scaled max-problem gap |
| `point` | `ndarray` shape `(2,)` | Attained `(prob, loss)` |
| `primal_dual_gap` | `float` | `|fun - (b_ub·λ + b_eq·ν)|` on the min problem |
| `max_stationarity_residual`, `max_complementarity_residual` | `float` | Dual residuals |
| `cash`, `forward` | `float` | Static cash and forward weights |
| `quote_positions` | `DataFrame` | Binding quotes |
| `structural` | `DataFrame` | Non-quote tight rows |
| `threshold_atom` | `float` | Atom mass at `K` |

`quote_positions` columns: `quote_id`, `cp_flag`, `strike`, `side`
(`bid`/`ask`), `weight`, `price`, `norm_price`. Bid weights are **signed**
(negative = sell at bid) so
`cash + forward + Σ weight * norm_price` matches `support_value` on hard
systems. Rows are kept when `|λ| > 1e-8` or the inequality residual is
`< 1e-8`.

`structural` columns: `constraint`, `weight`.

```python
cert.to_frame() -> pd.DataFrame  # kind, label, weight
```

Stacked rows: `cash`, `forward`, each quote (`"{quote_id} {side}"`), each
structural constraint, then `threshold_atom`.

`write_frontier(..., certificate=cert)` writes that frame as
`{stem}_binding.csv` and as `certificate` in the JSON bundle.

## Other return shapes

### scenario_check

```python
{
    "feasible": bool,
    "geometry": "inside" | "outside" | "boundary",
    "method": "lp" | "socp",
    "frontier": Frontier,
}
```

`inside` is immediately feasible; `outside` is immediately infeasible;
`boundary` is re-solved with an LP (`gamma=0`) or SOCP (`gamma>0`).

### portfolio_bound

`tuple[float, float]` — normalized price interval `(-h(-u), h(u))` for
payoff `u_p 1{X≤K} + u_L (K-X)^+`. Uses the **primal** support value, not
the duality-gap envelope. `gamma` other than `0` raises `ValueError`.
Support failure raises `RuntimeError`, not `SolverFailure`.

### nested_frontiers

`dict[str, Frontier]` keyed by quote-set name, in `sets` order. Default
`sets`: `put_pair`, `local_eight`, `put_wing` at `gamma=0`; those plus
`full_otm` at `gamma>0`.

## Export {#export}

```python
from crashfrontier import write_frontier
from crashfrontier.export import frontier_frames  # not a package-root name

frames = frontier_frames(fronts)          # dict of four DataFrames
paths = write_frontier(fronts, "outputs/f1", formats=("csv", "json"), certificate=None)
```

`fronts` is one `Frontier` or a `dict[str, Frontier]`. A single object is
named by its `quote_set`.

| Output | Contents |
|---|---|
| `{stem}_summary.csv` | Summary rows + `name` |
| `{stem}_vertices.csv` | Vertices + `name` |
| `{stem}_halfplanes.csv` | Half-planes + `name` |
| `{stem}_quotes.csv` | Program quotes + `name` |
| `{stem}_binding.csv` | Only if `certificate` is passed |
| `{stem}.json` | See below |

`formats` may be any subset of `csv` / `json`. Parent directories are
created. Return value is the list of written `pathlib.Path`s.

JSON bundle:

```text
{
  "frontiers": { "<name>": <Frontier.to_dict()>, ... },
  "certificate": null | [ {kind, label, weight}, ... ]
}
```

### Frontier.to_dict() {#todict}

JSON-serializable copy of the dataclass:

- `date` / `expiry` → `"YYYY-MM-DD"`
- `inner` / `outer` → lists of `[prob, loss]`
- `halfplanes` → lists of `[a_prob, a_loss, h]`
- `lo` / `hi` → two-element lists
- `normalizer` → `{"origin": [p, L], "scale": [dp, dL]}`
- `quotes` → list of row dicts
- `diagnostics` → JSON-safe dict (`Timestamp` → date string, NumPy → lists)
- non-finite numbers → JSON `null`; exported files use strict RFC JSON

`p0` is included. `summary()` keys that are derived (`prob_lo`,
`solver_gap`, …) are **not** duplicated; recompute them from the payload
or keep the CSV.

Related pages: [Python API](api.md), [configuration](configuration.md),
[exceptions](errors.md), [user-guide export](../user-guide/results-export.md).
