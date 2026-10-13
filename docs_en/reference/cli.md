# Command-line appendix

The entry point is `crashfrontier` (`crashfrontier.cli:main`). It calls the
same objects as the [Python API](api.md): `Dataset`, `compute_frontier`,
`nested_frontiers`, `scenario_check`, `binding_quotes`, `term_structure`,
`frontier_series`, `write_frontier`.

Result schemas: [results](results.md). Exceptions: [exceptions](errors.md).
Solver settings are **not** exposed as flags; use the Python API
([configuration](configuration.md)).

On error the CLI prints `error: ...` to stderr and exits with code `1`.
`CrashFrontierError`, `ImportError`, `ValueError`, and `OSError` are caught.
`--log-level DEBUG` also prints a traceback.

## Subcommands

| Command | API counterpart | Role |
|---|---|---|
| `inspect` | `Dataset.describe()` | Print recognized files, date format, and strike scale |
| `index` | date-byte index | Build or refresh the index for text option tables; `--force` rebuilds |
| `expiries DATE` | `Dataset.expiries` | List expiries and settlements for a date; `--lm-range LO HI` is available |
| `frontier` | `compute_frontier` / `nested_frontiers` | Compute and, optionally, plot and export |
| `scenario` | `scenario_check` | Check a `(prob, loss)` pair; `--prob` and `--loss` are required |
| `binding` | `binding_quotes` | Print the dual certificate; risk-neutral only |
| `term` | `term_structure` | Same observation date, several target tenors |
| `series` | `frontier_series` | Time series over a date range |

There is no `threshold` subcommand; use `threshold_profile` in Python.

```bash
crashfrontier inspect --data /path/to/market-data
crashfrontier index --data /path/to/market-data
crashfrontier expiries 2024-01-17 --data /path/to/market-data
crashfrontier frontier --data /path/to/market-data --date 2024-01-17 --target 30 --K 0.9 --out outputs/f1 --export csv,json
```

`inspect` on a large unindexed text file still prints a partial summary, then
`date index does not exist; run crashfrontier index` (or the stale variant),
and exits `0`. Other commands raise that hint as `error:` and exit `1`.

## Global options

Every subcommand accepts these flags. `--data` and `--options` cannot be
given together. If `--data` is omitted and `DATA/` exists in the current
directory, that directory is used. Public examples pass `--data` explicitly.

| Option | Meaning |
|---|---|
| `--data DIR` | Data directory (including an optional `RAW_DATA/`) via `Dataset.from_directory` |
| `--options PATH ...` | Option-table paths, one or more; a directory is scanned for option tables only (no `RAW_DATA/` walk) |
| `--forwards` / `--rates` / `--spot` | Forwards, rates, and spot. `--rates` may be a path or a numeric constant (already decimal) |
| `--strike-scale` | `auto` (default) or a positive scale |
| `--dayfirst` | Ambiguous dates are day/month/year |
| `--rate-unit` | `percent` (default, divide file/frame rates by 100) or `decimal` |
| `--column NAME=RAW` | Canonical column name to raw column name; repeatable |
| `--cache DIR` | **Date-index** directory. If omitted, the CLI uses `<first-option-file>/../../CACHE` |
| `--no-volume-filter` | Same as `QuoteFilter.permissive()` (`min_volume=0`) |
| `--min-volume` | Override the volume floor (applied after `--no-volume-filter` if both are set) |
| `--log-level` | Default `INFO` |

Text option files whose combined size exceeds `512 * 1024 * 1024` bytes
require a current date index (`crashfrontier index`).

## Selecting a cross-section

`frontier`, `scenario`, and `binding` must specify **exactly one** of
`--expiry`, `--days`, `--target`. `--window LO HI` can only be used with
`--target`. `--am-settlement` is `0` or `1`.

## Computation flags

Model flags on `frontier`:

| Option | Notes |
|---|---|
| `--K` | Threshold, default `0.90` |
| `--gamma` | `0`, `1`, or `2`, default `0` |
| `--support-cap CAP` | `none` / `null` / `unbounded` mean no upper bound; if omitted, `gamma>0` uses `5` and `gamma=0` is unbounded |
| `--quote-set` | Mutually exclusive with `--nested` |
| `--nested` | Calls `nested_frontiers` |
| `--sets` | Comma-separated nested set names; **must** be used with `--nested` |
| `--method` | `auto` / `hull` / `grid` |
| `--mode` | `auto` / `hard` / `relaxed` (slack fallback exists only for `gamma=0`) |
| `--p0` | Probability cutoff for transformed area and conditional-depth plots, default `1e-4` |
| `--layout` | `panels` (default) or `report`; the dual certificate is computed only when `--gamma 0` and `report` |
| `--out` | Output stem |
| `--formats` | Figure formats, default `pdf,png` |
| `--export` | For example `csv,json`; **must** be given together with `--out` |

`--export` on `frontier` calls `write_frontier` ([schemas](results.md#export)).
On `term`, `csv` writes the tenor table as `{out}.csv` (not the four frontier
CSVs); `json` writes `write_frontier(..., formats=("json",))` for the
successful frontiers.

`--tenors` on `term` defaults to `7,14,30,60,90`. `--freq` on `series` is
`D` / `W` / `M` / `Q`; `--target` defaults to `30`; `--resume` is the
`frontier_series` CSV checkpoint, **not** the `--cache` index directory.
`--dates` cannot be mixed with `--start` / `--end`; if `--dates` is omitted,
both endpoints are required. `--window` cannot be used with `--days`.
`series` always writes `{out}.csv` plus `{out}_prob`, `{out}_loss`, and
`{out}_depth` figures. `--overlay` is an optional spot-table path drawn on
the right axis.

`binding --direction` is `P,L` (default `1,0`).

## Correspondence with Python

```python
# The following CLI
# crashfrontier frontier --data /path/to/market-data --date 2024-01-17 --target 30 --K 0.9 --nested

from crashfrontier import Dataset, nested_frontiers

ds = Dataset.from_directory("/path/to/market-data")
cs = ds.cross_section("2024-01-17", target_days=30)
fronts = nested_frontiers(cs, K=0.9)
```

There are no command-line flags for a custom `QuoteFilter` beyond volume,
an in-memory `DataFrame`, or `LPSettings` / `ConicSettings` / `GridSettings`.
