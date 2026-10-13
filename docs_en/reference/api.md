# Python API

`import crashfrontier` re-exports the names below, matching
`crashfrontier.__all__`. Internal helpers are omitted.

Schemas and status tokens: [results](results.md). Exception types:
[exceptions](errors.md). Solver dataclasses: [configuration](configuration.md).
CLI flags: [command-line appendix](cli.md). File formats:
[data guide](../data-guide/index.md).

```python
from crashfrontier import (
    Dataset, CrossSection, QuoteFilter, ColumnMap,
    Frontier, compute_frontier, nested_frontiers,
    scenario_check, portfolio_bound, binding_quotes,
    term_structure, threshold_profile, frontier_series,
    plot_frontier, plot_conditional_depth, plot_frontier_panels,
    plot_quotes, plot_binding_quotes, plot_report,
    plot_term_structure, plot_threshold_profile, plot_time_series,
    save_figure, write_frontier,
    LPSettings, ConicSettings, GridSettings,
)
```

## Version {#version}

`crashfrontier.__version__` is the string `"0.1.0"`.

!!! warning "Not package-root names"
    Import `DualCertificate` from `crashfrontier.riskneutral`. Import
    `InfeasibleCrossSectionError`, `SolverFailure`, `SchemaError`,
    `NoSuchDateError`, `NoEligibleExpiryError`, `MissingForwardError`, and
    `CrashFrontierError` from `crashfrontier.errors`. `pick_dates` is
    `crashfrontier.series.pick_dates`. `frontier_frames` is
    `crashfrontier.export.frontier_frames`.

## Minimal path {#quickstart}

```python
from crashfrontier import Dataset, compute_frontier, plot_frontier_panels, save_figure

ds = Dataset.from_directory("/path/to/market-data")
date = ds.available_dates()[-1]
cs = ds.cross_section(date, target_days=30)
frontier = compute_frontier(cs, K=0.90)
print(frontier.describe())

fig = plot_frontier_panels(frontier)
save_figure(fig, "outputs/frontier")
```

With no files, use `CrossSection.from_quotes` and pass `forward` yourself.

## Data {#data}

### Dataset

Entry point for options / forwards / rates / spot. `options` may be a path,
a list of paths, a directory of option tables, or a `DataFrame`.

A directory passed to the **constructor** is scanned for option files only
and does **not** enter `RAW_DATA/` or attach forwards/rates/spot. Use
`from_directory` for mixed folders.

```python
Dataset(
    options,
    *,
    forwards=None,
    rates=None,
    spot=None,
    column_map: ColumnMap | dict[str, str] | None = None,
    strike_scale: float | str = "auto",
    dayfirst: bool | None = None,
    cp_values: dict[str, str] | None = None,
    rate_unit: str = "percent",
    forward_sources: tuple[str, ...] = ("table", "column", "parity"),
    quote_filter: QuoteFilter = QuoteFilter(),
    cache_dir: str | Path | None = None,
    memory_limit: int = 512 * 1024 * 1024,
) -> Dataset
```

| Parameter | Default | Notes |
|---|---|---|
| `forwards` / `rates` / `spot` | `None` | Path, frame, or (rates) a numeric constant |
| `strike_scale` | `"auto"` | `"auto"` or a positive number. `auto` chooses among `1, 10, 100, 1000` so the median strike is close to a reference price (forward table, then `forward_price` column, then spot). With no reference and median strike `> 1e5`, scale `1000` is used |
| `rate_unit` | `"percent"` | File/frame values are divided by 100. A numeric `rates=0.02` is already decimal and **ignores** this |
| `forward_sources` | `("table", "column", "parity")` | First finite value wins. Unknown names raise `SchemaError` |
| `memory_limit` | `512 * 1024 * 1024` | Large text files use a date-byte index under `cache_dir` |
| `dayfirst` | `None` | `True` prefers day/month/year on ambiguous dates |

`rates=None` uses a zero rate (`rate_source="zero"`) and logs a warning on
first use. `rates=True`/`False` raises `SchemaError`.

| Method | Signature / role | Raises |
|---|---|---|
| `from_directory(root, **kwargs)` | Classify one-level files under `root` and optional `RAW_DATA/`. Unrecognized tables are ignored. A filename containing `vix` or `vx` is not treated as spot | `SchemaError` if `root` is not a directory, no option tables, or several forwards/rates/spot files |
| `from_frames(options, *, forwards=None, rates=None, spot=None, **kwargs)` | In-memory tables | Same as the constructor |
| `from_spec(spec)` | Rebuild a file-backed dataset | Same as the constructor |
| `spec()` → `dict` | JSON-serializable constructor args, including the **resolved** `strike_scale` | `ValueError` if any input was a `DataFrame` |
| `describe()` → `dict` | Files, date format, strike scale, auxiliaries | `SchemaError` if the source has no dates |
| `available_dates()` → `list[pd.Timestamp]` | Sorted observation dates | — |
| `nearest_date(date, direction="backward")` → `pd.Timestamp` | `backward` or `forward` | `ValueError` on a bad direction; `NoSuchDateError` if the walk leaves the range |
| `option_rows(date)` → `pd.DataFrame` | Canonical rows for that date | `NoSuchDateError`; `SchemaError` on an unparseable date |
| `expiries(date, lm_range=(-0.35, 0.05))` → `pd.DataFrame` | One row per `(expiry, am_settlement, root)` | `NoSuchDateError` / `SchemaError` as above. Missing forwards become `forward=nan`, `forward_source="none"` |
| `cross_section(date, expiry=None, *, days=None, target_days=None, window=None, am_settlement=None, root=None, min_put_strikes=0, lm_range=(-0.35, 0.05), forward=None, rate=None)` → `CrossSection` | At least one of `expiry`, `days`, `target_days`; if several are passed, precedence is `expiry` → `days` → `target_days`. `window=(lo, hi)` applies only with `target_days` (else the filter day limits). Ties: nearest `days`, then earliest expiry, then more put strikes, then PM | `ValueError` if none of the three selectors is set; `NoEligibleExpiryError` if nothing remains (including expiries whose forward chain failed); `SchemaError` |
| `cross_sections(date, **kwargs)` → `list[CrossSection]` | Every eligible `(expiry, settlement, root)` on `date`. Honors `lm_range`, `am_settlement`, `root`, `forward`, `rate`. Ignores `expiry` / `days` / `target_days` / `window` / `min_put_strikes` | Empty list if nothing qualifies |

User-supplied `forward` / `rate` set `forward_source` / `rate_source` to
`"user"`. Ambiguous AM/PM without `am_settlement` logs a warning and sets
`CrossSection.ambiguous_settlement=True`.

`Dataset` is not a dataclass; do not construct `CrossSection` fields from it
by hand.

### CrossSection

Frozen dataclass: one observation date, one expiry, one settlement.
Usually from `Dataset.cross_section` or `from_quotes`.

```python
CrossSection.from_quotes(
    quotes: pd.DataFrame,
    *,
    date,
    expiry,
    forward: float,
    rate: float = 0.0,
    spot: float = nan,
    am_settlement: int = 0,
    column_map: ColumnMap | None = None,
    strike_scale: float = 1.0,
    dayfirst: bool | None = None,
    cp_values: dict[str, str] | None = None,
    quote_filter: QuoteFilter = QuoteFilter(),
) -> CrossSection
```

`strike_scale` here defaults to `1.0`, not `"auto"`. `column_map` is
`ColumnMap` or `None` (not a `dict`). If `column_map` is omitted, missing
date/expiry columns are filled from the arguments and `ColumnMap.detect`
is used. `forward_source` and `rate_source` are `"user"`.

`tau` is `days / 365`. `discount` is `exp(-rate * tau)`.

| Member | Type | Role |
|---|---|---|
| `quotes` | `pd.DataFrame` | Normalized panel; [columns](results.md#quotes) |
| `puts` / `calls` | `pd.DataFrame` | Slices by `cp_flag` |
| `filter_report` | `dict[str, int]` | Per-rule drop counts plus `kept` |
| `summary()` | `dict[str, object]` | Identifiers and quote counts |

Raises `SchemaError` from column detection / `cp_flag` parsing.

### QuoteFilter

```python
QuoteFilter(
    min_volume: float = 1.0,
    min_open_interest: float = 0.0,
    positive_strike: bool = True,
    nonnegative_bid: bool = True,
    positive_ask: bool = True,
    uncrossed: bool = True,
    min_days: int = 7,
    max_days: int = 120,
    min_width: float = 0.0,
    require_finite: bool = True,
    cp_flags: tuple[str, ...] = ("C", "P"),
    one_quote_per_strike: bool = False,
)
QuoteFilter.default() -> QuoteFilter      # same as QuoteFilter()
QuoteFilter.permissive() -> QuoteFilter   # only min_volume=0.0
```

```python
filter.apply_rows(rows) -> tuple[pd.DataFrame, dict[str, int]]
filter.apply_normalized(quotes) -> tuple[pd.DataFrame, dict[str, int]]
```

`apply_rows` expects canonical columns including `days`. If `volume` is
absent or all-null, the volume rule is skipped and
`volume_unavailable=1`. `one_quote_per_strike` keeps the narrowest
`(cp_flag, strike)` after normalization. Frozen dataclass.

### ColumnMap

Map raw headers to canonical option columns. Required fields: `date`,
`expiry`, `cp_flag`, `strike`, `bid`, `ask`.

```python
ColumnMap(
    date, expiry, cp_flag, strike, bid, ask,
    volume=None, open_interest=None, impl_volatility=None,
    quote_id=None, am_settlement=None, symbol=None, root=None,
    forward_price=None, expiry_indicator=None,
)
ColumnMap.detect(columns, overrides: dict[str, str] | None = None) -> ColumnMap
ColumnMap.optionmetrics() -> ColumnMap
map.usecols() -> list[str]
map.rename_map() -> dict[str, str]
```

Names are compared after stripping BOM and paired quotes, then replacing
spaces, `-`, and `.` with `_`. `optionmetrics()` maps `exdate`,
`strike_price`, `best_bid`, `best_offer`, `optionid`, and related names.
A missing required field or an unknown/unmatched override raises
`SchemaError`.

## Compute {#compute}

### compute_frontier

```python
compute_frontier(
    cs: CrossSection,
    K: float = 0.90,
    *,
    gamma: int = 0,
    quote_set: str | pd.DataFrame | None = None,
    design: str | None = None,
    support_cap: float | None | str = "auto",
    mode: str = "auto",
    budget: float | None = None,
    method: str = "auto",
    p0: float = 1e-4,
    normalizer=None,
    lp_settings: LPSettings | None = None,
    conic_settings: ConicSettings | None = None,
    grid_settings: GridSettings | None = None,
    full_strike_count: int | None = None,
) -> Frontier
```

| Parameter | Default | Behavior |
|---|---|---|
| `gamma` | `0` | `0` risk-neutral LP; `1` / `2` physical SOCP. Other values raise `ValueError` |
| `quote_set` | `None` | `None` → `put_wing` (`gamma=0`) or `full_otm` (`gamma>0`). A `DataFrame` is stored as `"custom"` |
| `design` | `None` | `None` → `riskneutral` (`gamma=0`) or `physical`. Risk-neutral names: `put_pair`, `local_eight`, `put_wing`, `full_panel`. Physical adds `full_otm`. The risk-neutral pair uses the nearest puts at-or-below / at-or-above \(K\); the physical pair uses distinct strikes strictly below / above \(K\). Unknown name/design raises `ValueError` |
| `support_cap` | `"auto"` | `"auto"` → `5.0` if `gamma>0`, unbounded if `gamma=0`. `None` / `"none"` / `"null"` / `"unbounded"` / non-finite → unbounded. At `gamma>0` a finite cap must exceed `max(1, K)` |
| `mode` | `"auto"` | `gamma=0` only. `auto` tries hard constraints, then the minimum slack budget `δ* + 1e-10` and logs a warning. `hard` does not fall back. `relaxed` with `budget < δ*` fails |
| `method` | `"auto"` | `"auto"` → hull (`gamma=0`) or grid (`gamma>0`). A marginal width `≤ 1e-10` (see `LPSettings.degenerate_tolerance`) yields `method="degenerate"`, area `0`, and undefined (`NaN`) `area_ratio` |
| `p0` | `1e-4` | Transformed-area / depth-boundary floor |
| `full_strike_count` | `None` | Slack weights for nested `gamma=0` layers; default is the distinct-strike count of the passed quotes |
| settings | `None` | Default instances; see [configuration](configuration.md) |

At `gamma>0`, `K` must satisfy `0 < K < 1` (`ValueError` otherwise).
Risk-neutral `K` need only be finite. Both models require at least two
distinct quote strikes (`ValueError`).

Raises `InfeasibleCrossSectionError` (fields `date`, `expiry`, `quote_set`,
`gamma`, `support_cap`, `status`) when no measure exists. `gamma>0` has no
slack fallback; the message may mention `put_wing` or `support_cap=None`.
Raises `SolverFailure` on a failed support solve. A physical depth-enclosure
discrepancy leaves the joint frontier intact and can set the public depth
endpoints to NaN. A finite risk-neutral support cap that empties the feasible
set raises `InfeasibleCrossSectionError`; another axis-bound LP
failure can still raise `RuntimeError` (`marginal support failed`).

### Frontier

Result object. Field list, units, statuses, and `to_dict()`:
[results](results.md#frontier).

```python
frontier.describe() -> str
frontier.summary() -> dict[str, object]
frontier.vertices() -> pd.DataFrame          # kind, order, prob, loss, depth
frontier.halfplane_table() -> pd.DataFrame   # a_prob, a_loss, h
frontier.contains(prob, loss) -> str         # inside / outside / boundary
frontier.depth_boundary(samples_per_edge=80, *, p_floor=None) -> np.ndarray
frontier.to_dict() -> dict
```

`contains`: inner polygon (including its boundary) is `inside`; strictly
outside the outer polygon is `outside`; the sandwich is `boundary`.
`summary()` column order is fixed; `solver_gap` is the max LP dual gap at
`gamma=0` and the max cone violation at `gamma>0`.

### nested_frontiers

```python
nested_frontiers(
    cs: CrossSection,
    K: float = 0.90,
    *,
    gamma: int = 0,
    sets=None,
    accumulate_halfplanes=None,
    **kwargs,
) -> dict[str, Frontier]
```

Default `sets`: `("put_pair", "local_eight", "put_wing")` at `gamma=0`;
those plus `"full_otm"` at `gamma>0`. An empty `sets` raises `ValueError`.
`quote_set` in `kwargs` is discarded. Remaining keywords go to
`compute_frontier`.

`gamma=0` and `mode="auto"`: hard feasibility is decided on the **largest**
set. If that set is infeasible, every layer uses the same relaxed budget
(largest-set `δ* + 1e-10`, or caller `budget`) and the same
`full_strike_count`.

`accumulate_halfplanes` defaults to `True` when `gamma>0` and `False`
otherwise. When true, each layer's inner and outer polygons are intersected
with half-planes accumulated up to that layer. The stored half-plane set and
all transformed-area summaries are updated.

An infeasible layer raises `InfeasibleCrossSectionError` (does not skip).
At `gamma>0`, a `SolverFailure` on a later layer may retry once without the
shared `put_pair` normalizer.

### scenario_check

```python
scenario_check(
    cs: CrossSection,
    K,
    prob,
    loss,
    *,
    gamma: int = 0,
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> dict
```

Returns `{"feasible": bool, "geometry": str, "method": "lp"|"socp",
"frontier": Frontier}`. `inside` is feasible; `outside` is not; `boundary`
is re-solved. Same exceptions as `compute_frontier`.

### portfolio_bound

```python
portfolio_bound(
    cs: CrossSection,
    K,
    u_prob,
    u_loss,
    *,
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> tuple[float, float]
```

Normalized price interval `(-h(-u), h(u))` for
`u_p 1{X≤K} + u_L (K-X)^+`. Defined only for `gamma=0` (`ValueError`
otherwise). Support failure raises `RuntimeError`, not `SolverFailure`.

### binding_quotes

```python
binding_quotes(
    cs: CrossSection,
    K,
    *,
    direction=(1.0, 0.0),
    quote_set: str | pd.DataFrame | None = None,
    **kwargs,
) -> DualCertificate
```

Static super-replicating certificate in a raw `(prob, loss)` direction.
`gamma` must be `0`. Constraint mode is always `"auto"` (`mode` in
`kwargs` is ignored). If the hard system is infeasible, the slack system is
used and a warning is logged. Support failure raises `SolverFailure`.
Import the type: `from crashfrontier.riskneutral import DualCertificate`.
Fields and `to_frame()`: [results](results.md#certificate).

## Batch computation {#series}

`term_structure` and `threshold_profile` never abort: a failed tenor or
threshold becomes an `error` cell (`CrashFrontierError` or `ValueError`).
`frontier_series` does the same when `on_error="skip"`.

### term_structure

```python
term_structure(
    ds: Dataset,
    date,
    *,
    tenors: Sequence[int] = (7, 14, 30, 60, 90),
    window_fraction: float = 0.35,
    K: float = 0.90,
    gamma: int = 0,
    am_settlement: int | None = None,
    **frontier_kwargs,
) -> tuple[pd.DataFrame, dict[int, Frontier]]
```

Same observation date, several target tenors. The window is
`round(tenor * (1 ± window_fraction))`, clipped to `QuoteFilter` day
limits. Duplicate expiries keep the tenor closer to its target; the other
row gets `error="expiry already used"`. Table columns: [summary](results.md#summary)
plus `target_days`, `error`. The dict is keyed by tenor and omits failures.

### threshold_profile

```python
threshold_profile(
    cs: CrossSection,
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
    *,
    gamma: int = 0,
    **kwargs,
) -> tuple[pd.DataFrame, dict[float, Frontier]]
```

Same cross-section, several `K`. Table columns: summary plus `error`. Dict
keyed by threshold.

### frontier_series

```python
frontier_series(
    ds: Dataset,
    dates: Sequence | None = None,
    *,
    start=None,
    end=None,
    freq: str = "W",
    K: float = 0.90,
    gamma: int = 0,
    quote_set=None,
    days: int | None = None,
    target_days: int | None = 30,
    window: tuple[int, int] | None = (20, 45),
    am_settlement: int | None = None,
    snap: str | None = "backward",
    workers: int = 1,
    cache_path: str | Path | None = None,
    on_error: str = "skip",
    **frontier_kwargs,
) -> pd.DataFrame
```

| Parameter | Constraint |
|---|---|
| `dates` | If `None`, sample `available_dates()` with `start` / `end` / `freq`. `freq` is `D`, `W`, `M`, `Q` (`ValueError` otherwise). Weekly dates prefer Wednesday, Tuesday, Thursday, Monday, Friday; month/quarter take the last date in the period that has data |
| `days` vs `target_days` | If `days` is set, `target_days` and `window` are ignored |
| `snap` | `backward`, `forward`, or `None`. Exact matches are used as-is. `None` raises `NoSuchDateError` when the requested date is missing |
| `on_error` | `skip` or `raise` |
| `workers` | `>= 1`. `>1` rebuilds via `spec()`; an in-memory source logs a warning and uses one process |
| `cache_path` | CSV checkpoint of **series rows**, not `Dataset(cache_dir=...)` |

Extra columns: `requested_date`, `error`. Empty `dates` returns an empty
table with those columns. Sampling helper (not package-root):
`crashfrontier.series.pick_dates`.

## Plotting {#plotting}

All functions return `matplotlib.figure.Figure`. `fronts` may be one
`Frontier`, a name→object mapping, or a sequence (empty mapping/sequence
raises `ValueError`).

```python
plot_frontier(fronts, *, ax=None, fill=True, outline_outer=True,
              show_bounds=False, loss_unit="bp", annotate=True) -> Figure
plot_conditional_depth(fronts, *, ax=None, log_p=None, inset=True, legend=True) -> Figure
plot_frontier_panels(fronts, *, title=None) -> Figure
plot_quotes(cs, K, *, ax=None, highlight=None, used=None) -> Figure
plot_binding_quotes(cert, *, ax=None) -> Figure
plot_report(cs, fronts, *, certificate=None, title=None) -> Figure
plot_term_structure(table, *, ax=None) -> Figure
plot_threshold_profile(table, *, ax=None) -> Figure
plot_time_series(table, *, value="prob", ax=None, overlay=None, overlay_label=None) -> Figure
save_figure(fig, path_stem, formats=("pdf", "png"), dpi=300) -> list[Path]
```

`loss_unit="bp"` selects basis points. 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%.

| Parameter | Constraint |
|---|---|
| `loss_unit` | `bp` or `percent` (`ValueError` otherwise). The name stays `loss_unit="bp"` |
| `value` | `prob`, `loss`, or `depth` (`ValueError` otherwise) |
| `highlight` / `certificate` | `DualCertificate` or `None` |
| `used` | Optional quote subset; other quotes are drawn faded |
| `log_p` | `None` selects a log probability axis when the displayed range spans at least a factor of 12 |

On figures, probability and depth are percents. Default loss axes are basis
points: 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%.
The axis text spells out “basis points”; the conversion is documented here
and in [Read the figure](../getting-started/read-the-chart.md#coordinates-and-units).
`loss_unit="percent"` draws loss as a percent instead, still from the same
stored decimal (`×100` rather than `×10000`). Rows with a nonempty `error`
are not drawn. Styles are applied on each figure; importing the plot
module does not change global Matplotlib settings.
`save_figure` creates parent directories. If `path_stem` already ends with
`.pdf`, `.png`, `.svg`, `.eps`, `.jpg`, or `.jpeg`, that suffix is stripped
before `formats` are appended.

Stored decimals stay decimals; see [units](results.md#units).

## Export {#export}

```python
write_frontier(
    fronts: Frontier | dict[str, Frontier],
    path_stem: str | Path,
    *,
    formats: tuple[str, ...] = ("csv", "json"),
    certificate: DualCertificate | None = None,
) -> list[Path]
```

Writes `{stem}_summary.csv`, `{stem}_vertices.csv`, `{stem}_halfplanes.csv`,
`{stem}_quotes.csv` when `"csv"` is requested; `{stem}_binding.csv` if
`certificate` is passed; `{stem}.json` when `"json"` is requested. A single
`Frontier` is keyed by `quote_set`. Schemas: [results](results.md#export).

In-memory tables (not package-root): `crashfrontier.export.frontier_frames`.

## Numerical settings {#settings}

```python
LPSettings(
    primal_feasibility_tolerance=1e-10,
    dual_feasibility_tolerance=1e-10,
    time_limit_seconds=5.0,
    edge_tolerance=1e-8,
    point_tolerance=1e-9,
    degenerate_tolerance=1e-10,
    max_support_calls=500,
    anytime_max_support_calls=160,
    anytime_area_relative_tolerance=0.002,
    anytime_area_absolute_tolerance=1e-12,
    interior_cutoff=1e-8,
    validation_multiplier=5.0,
)
GridSettings(levels=(32, 64, 128), max_normalized_gap=0.01)
ConicSettings(
    solver="CLARABEL",
    fallbacks=("ECOS", "SCS"),
    tolerance=1e-9,
    max_iterations=2000,
    violation_tolerance=1e-6,
    objective_tolerance=1e-7,
    warm_start=True,
)
```

When omitted, `compute_frontier` uses these default instances. Field-by-field
behavior, HiGHS/CVXPY fallbacks, and the SCS hard-coded tolerances:
[configuration](configuration.md).
