# Exceptions

Public exception types live in `crashfrontier.errors`. They are **not** in
`crashfrontier.__all__`. Catch the base class for any package error; catch a
subclass when the recovery action is specific.

```python
from crashfrontier.errors import (
    CrashFrontierError,
    InfeasibleCrossSectionError,
    MissingForwardError,
    NoEligibleExpiryError,
    NoSuchDateError,
    SchemaError,
    SolverFailure,
)
```

`term_structure`, `threshold_profile`, and `frontier_series(..., on_error="skip")`
also swallow `ValueError` (and `CrashFrontierError`) per item. Other public
functions let `ValueError` and, in two risk-neutral helpers, `RuntimeError`
propagate. See [untyped escapes](#untyped-escapes).

## Hierarchy

```text
Exception
└── CrashFrontierError
    ├── SchemaError
    ├── NoSuchDateError
    ├── NoEligibleExpiryError
    ├── MissingForwardError
    ├── InfeasibleCrossSectionError
    └── SolverFailure
```

All subclasses take a message string via `Exception`. Only
`InfeasibleCrossSectionError` adds fields.

## CrashFrontierError

```python
CrashFrontierError(*args)
```

Base class. No extra attributes. Import: `crashfrontier.errors`.

Raised only through subclasses in this package. Batch helpers treat any
`CrashFrontierError` as a per-item failure when they are in skip mode.

## SchemaError

```python
SchemaError(*args)
```

Input columns, files, or values do not match the expected schema. Import:
`crashfrontier.errors`. Attributes: message only.

Typical causes:

| Call | Condition |
|---|---|
| `ColumnMap.detect` | Required field missing; unknown override name; override column not in the header |
| `Dataset(...)` / `from_directory` | Not a directory; no option tables; mixed/ambiguous forwards, rates, or spot files; `strike_scale` is a string other than `"auto"`; unknown `forward_sources` name; `column_map` is neither `ColumnMap` nor `dict`; `rates=True`/`False` |
| `Dataset.option_rows` / `cross_section` | Date string cannot be parsed |
| Readers / rates / forwards / spot | Unrecognized file type, `cp_flag`, settlement, numeric date, or rate unit |

`rate_unit` must be `"percent"` or `"decimal"`.

## NoSuchDateError

```python
NoSuchDateError(*args)
```

The requested observation date is not in the option source. Import:
`crashfrontier.errors`. Attributes: message only.

| Call | Condition |
|---|---|
| `Dataset.option_rows(date)` | `date` parses but has no rows; message includes nearest available dates |
| `Dataset.nearest_date` | Source has no dates, or `direction="backward"`/`"forward"` walks off the range |
| `frontier_series(..., snap=None)` | Requested date is not an option date |

`nearest_date` does **not** raise when `date` itself is present.

## NoEligibleExpiryError

```python
NoEligibleExpiryError(*args)
```

No expiry survives the selection rules for that date. Import:
`crashfrontier.errors`. Attributes: message only.

Raised by `Dataset.cross_section` when `expiry`, `days`, or `target_days`
matches nothing after filtering to finite forwards, `QuoteFilter.min_days` /
`max_days`, optional `am_settlement` / `root`, and (for `target_days`) the
window and `min_put_strikes`. The message includes `expiries(date)` as a
table.

`Dataset.cross_sections` does not raise this: it returns `[]` when there are
no candidates.

## MissingForwardError

```python
MissingForwardError(*args)
```

No source in `forward_sources` returned a finite forward. Import:
`crashfrontier.errors`. Attributes: message only.

Raised by the forward resolver when every configured source fails.
`Dataset.expiries` **catches** it and writes `forward=nan`,
`forward_source="none"`. `Dataset.cross_section` then drops that expiry
(candidates require a finite forward) and raises `NoEligibleExpiryError`
if nothing remains.

## InfeasibleCrossSectionError

```python
InfeasibleCrossSectionError(
    message: str = "",
    *,
    date=None,
    expiry=None,
    quote_set: str | None = None,
    gamma: int | None = None,
    support_cap: float | None = None,
    status: str | None = None,
)
```

Quotes admit no feasible measure under the chosen model. Import:
`crashfrontier.errors`.

| Attribute | Type in practice | Meaning |
|---|---|---|
| `date` | `pd.Timestamp` or `None` | Observation date |
| `expiry` | `pd.Timestamp` or `None` | Expiry |
| `quote_set` | `str` or `None` | Set name (`custom` for a passed `DataFrame`) |
| `gamma` | `int` or `None` | `0`, `1`, or `2` |
| `support_cap` | `float` or `None` | Resolved cap; `None` is unbounded |
| `status` | `str` or `None` | Solver or reconstruction token |

`compute_frontier` fills every field. `gamma=0` uses `status="infeasible"`
when `mode="hard"` is infeasible, `mode="relaxed"` cannot meet `budget`, or
`mode="auto"` cannot build a slack system. `gamma>0` copies the physical
feasibility status (for example a CVXPY `infeasible` token). There is **no**
slack fallback at `gamma>0`. The message may mention retrying
`quote_set="put_wing"` or `support_cap=None`; neither is guaranteed.

`nested_frontiers` re-raises at the first failed layer. The new message
lists completed set names. Fields are copied from the inner error;
`quote_set` is the failed layer.

A bare constructor (`InfeasibleCrossSectionError("...")`) leaves the extra
fields at `None` and `gamma` uncoerced.

```python
from crashfrontier.errors import InfeasibleCrossSectionError

try:
    compute_frontier(cs, 0.90, gamma=2)
except InfeasibleCrossSectionError as exc:
    print(exc.quote_set, exc.support_cap, exc.status)
```

## SolverFailure

```python
SolverFailure(*args)
```

An LP or conic support solve did not return a usable solution. Import:
`crashfrontier.errors`. Attributes: message only.

Raised when:

- a support oracle used by hull/grid reconstruction fails
- physical marginal bounds are non-finite after a feasible certificate
- `binding_quotes` support fails

The message includes the raw `(prob, loss)` direction or the
`gamma` / `quote_set` pair.

## Untyped escapes {#untyped-escapes}

These are not `CrashFrontierError` subclasses but do leave public functions.

| Exception | Public call | Typical message / cause |
|---|---|---|
| `ValueError` | `compute_frontier` | `unsupported gamma`; `unrecognized reconstruction method`; `unrecognized constraint mode`; `unknown quote set`; `unknown design`; `at least two distinct quote strikes are required`; `K must lie strictly between zero and one` (`gamma>0`); `support_cap must be finite and exceed max(1, K)` (`gamma>0`) |
| `ValueError` | `nested_frontiers` | `sets must contain at least one quote-set name`; same as `compute_frontier` |
| `ValueError` | `portfolio_bound` / `binding_quotes` | `... is defined for gamma=0` |
| `ValueError` | `Dataset.cross_section` | `cross_section requires expiry, days, or target_days` |
| `ValueError` | `Dataset.nearest_date` | `direction must be 'backward' or 'forward'` |
| `ValueError` | `Dataset.spec` | `in-memory option source cannot be serialized` |
| `ValueError` | `frontier_series` | `on_error`, `snap`, `workers`, or `freq` |
| `ValueError` | `plot_time_series` | `value must be 'prob', 'loss', or 'depth'` |
| `ValueError` | `plot_frontier` | `loss_unit must be 'bp' or 'percent'` |
| `ValueError` | empty `fronts` mapping/sequence in plotting | `fronts is empty` |
| `RuntimeError` | `portfolio_bound` | `portfolio support failed` |
| `RuntimeError` | `compute_frontier` (`gamma=0`) | `marginal support failed` (from the axis-bound LPs) |

`portfolio_bound` does **not** raise `SolverFailure`. `binding_quotes` does.

Batch tables record `CrashFrontierError` and `ValueError` in the `error`
column. `RuntimeError` from a worker is not converted to an `error` row.

## Error-to-action {#lookup}

| Symptom | Type | First action |
|---|---|---|
| Missing `date` / `expiry` / `cp_flag` / `strike` / `bid` / `ask` | `SchemaError` | Pass `column_map=` or `--column NAME=RAW`; see [data guide](../data-guide/formats-columns.md) |
| `Dataset(path)` finds no options when files are under `RAW_DATA/` | `SchemaError` | Use `Dataset.from_directory(path)` so the optional `RAW_DATA/` level is scanned |
| Several forward/rate/spot tables | `SchemaError` | Pass `forwards=`, `rates=`, or `spot=` explicitly |
| Date not in the file | `NoSuchDateError` | `ds.nearest_date(date)` or `snap="backward"` on a series |
| `days=2` / empty window | `NoEligibleExpiryError` | `ds.expiries(date)` and widen `window` or `QuoteFilter` day limits |
| No finite forward | `NoEligibleExpiryError` (`MissingForwardError` is swallowed by `expiries`) | Supply a forward table or a `forward_price` column |
| `gamma=0` and `quote_set="full_otm"` | `ValueError` | `full_otm` exists only for `design="physical"`; default risk-neutral names are `put_pair`, `local_eight`, `put_wing`, `full_panel` |
| `mode="hard"` on crossed quotes | `InfeasibleCrossSectionError` | `mode="auto"` (slack budget) or drop the conflicting quotes |
| Default physical `full_otm` + `support_cap=5` | `InfeasibleCrossSectionError` | Retry `quote_set="put_wing"` and/or `support_cap=None` |
| Nested physical run dies at `full_otm` | `InfeasibleCrossSectionError` | Pass `sets=("put_pair", "local_eight", "put_wing")` so `full_otm` is omitted |
| `K` is `0` or `1` at `gamma>0` | `ValueError` | Physical models require `0 < K < 1` |
| `support_cap=1.0` at `gamma>0` | `ValueError` | Cap must exceed `max(1, K)` |
| Support / cone failure mid-reconstruction | `SolverFailure` | Retry `method="grid"` or looser `GridSettings`; install `crashfrontier[solvers]` for ECOS/SCS |
| `workers>1` on `Dataset.from_frames` | warning, then `workers=1` | File-backed `Dataset` so `spec()` can pickle paths |
| One tenor/date fails in a batch | `error` column | Filter `table[table["error"].eq("")]`; the call does not abort unless `on_error="raise"` |

Related pages: [Python API](api.md), [results](results.md),
[configuration](configuration.md), [troubleshooting](../user-guide/troubleshooting.md).
