# Data guide

How to turn an **OptionMetrics export** or **your own quote table** into a
`Dataset` / `CrossSection` that `crashfrontier` can compute on. Behavior is
taken from `crashfrontier/data/` and `tests/test_*.py`.

Assumes an importable install (the base [install](../getting-started/install.md) is enough). CLI
(`inspect` / `index`) appears only where it matches a Python check.

How to point the library at files you already hold:
[Market data](../getting-started/data.md). Scripts that open a real extract
should read the folder from `CRASHFRONTIER_DATA`. Copy-paste directory
examples on these pages use `/path/to/market-data` or a small table in a
temporary directory.

## There is no `Dataset.from_files`

`hasattr(Dataset, "from_files")` is `False`. That name is not on the class.

| What you have | Use | Do not use |
|---|---|---|
| A directory that (or whose `RAW_DATA/`) mixes options, forwards, rates, and spot | [`Dataset.from_directory`](from-directory.md) | `Dataset("/path/to/market-data")`, `Dataset.from_files(...)` |
| Known file paths | [`Dataset(...)` constructor](from-directory.md#constructor-dataset-is-not-from_directory) | Expecting the constructor to scan `RAW_DATA/` or attach forwards/rates |
| Multi-day pandas tables | [`Dataset.from_frames`](from-frames.md) | Passing a DataFrame to the CLI |
| One observation date, one expiry | [`CrossSection.from_quotes`](from-quotes.md) | Editing bid/ask so a solve “goes through” |

## Reading order

1. Overview: entry points and the canonical schema.
2. [Formats and column aliases](formats-columns.md): suffixes, delimiters, `ColumnMap.detect`.
3. [Dates and strike scaling](dates-strikes.md): `dayfirst`, `strike_scale`.
4. Entry points: [directory](from-directory.md) · [in-memory frames](from-frames.md) · [single-day single-expiry](from-quotes.md).
5. [Forwards and rates](forwards-rates.md), [quote filter](filters.md), [byte-offset date index](large-files.md).
6. Walk the [validation checklist](validation-checklist.md) before `compute_frontier`.

## Canonical option rows

`Dataset.option_rows(date)` and `canonicalize(...)` emit the same columns.
Required mappings: `date`, `expiry`, `cp_flag`, `strike`, `bid`, `ask`. A
missing required field makes `ColumnMap.detect` raise `SchemaError` with
aliases and the raw header.

| Canonical name | Type | Required | Rule |
|---|---|---|---|
| `date` | midnight `datetime64[ns]` | yes | observation date |
| `expiry` | midnight | yes | expiry |
| `cp_flag` | `"C"` / `"P"` | yes | see `parse_cp_flag` |
| `strike` | float | yes | **already divided by** `strike_scale` |
| `bid` / `ask` | float | yes | same price unit as the underlying |
| `volume` / `open_interest` / `impl_volatility` / `forward_price` | float | no | missing column → all NaN |
| `quote_id` | str | no | missing column → `r{row number}` |
| `am_settlement` | 0 or 1 | no | missing column → inferred from `root` |
| `symbol` / `root` / `expiry_indicator` | str | no | missing → `""` |
| `days` | int | derived | `(expiry - date).days` |

Common OptionMetrics source names: `exdate`, `strike_price`, `best_bid`,
`best_offer`, `optionid`. Full alias table: [formats-columns.md](formats-columns.md).

## Opening a directory you supply

`from_directory` classifies one level under the folder you pass and, if it
exists, one level under `RAW_DATA/`. It does not recurse. Replace the path
with your extract, or set `CRASHFRONTIER_DATA`.

```python
from pathlib import Path
from crashfrontier import Dataset

root = Path("/path/to/market-data")
# real-extract scripts: Path(os.environ["CRASHFRONTIER_DATA"])
assert not hasattr(Dataset, "from_files")

ds = Dataset.from_directory(root)
info = ds.describe()
print(info["options"]["kind"], info["options"]["n_dates"])
print(info["date_format"], info["strike_scale"])
print(info["forwards"], info["rates"], info["spot"])
```

`describe()` reports what **your** files produced: date range, strike scale,
and whether a forward table, rate series, and spot file were attached.
OptionMetrics IvyDB extracts typically
use `"%Y-%m-%d"` dates and `strike_scale` 1000; a FRED-style two-column rate
file classifies as `"series"`. How `kind` is chosen (for example
`indexed_text` when an uncompressed options CSV exceeds 512 MiB) is in
[large-files.md](large-files.md).

The constructor is not a substitute. It looks only at that folder: it does
not enter `RAW_DATA/`, and it does not attach forwards, rates, or spot.

```python
from crashfrontier.errors import SchemaError

try:
    Dataset(root)
except SchemaError as exc:
    print(exc)  # no option tables in directory ... if options sit only in RAW_DATA/
```

Rows dropped by the quote filter, `InfeasibleCrossSectionError`, or a missing
forward are not a reason to invent bid/ask. Read `filter_report`, check
`strike_scale` / dates / AM·PM / the forward source, pick another day, or
widen a **documented** quote filter and record it. See [filters.md](filters.md)
and [validation-checklist.md](validation-checklist.md).

## Public defaults (literals from source)

| Item | `Dataset` | `CrossSection.from_quotes` |
|---|---|---|
| `strike_scale` | `"auto"` (chosen once from 1/10/100/1000 on first row read, then cached) | `1.0` (**no** auto-detect) |
| `dayfirst` | `None` (ambiguous dates are month-day-year, with a warning) | same |
| `rate_unit` | `"percent"` (two-column rate series only) | none; `rate=` is already a decimal |
| `rate` | from the rate source | `0.0`, `rate_source="user"` |
| `forward` | chain `("table", "column", "parity")` | required argument, `forward_source="user"` |
| `am_settlement` | from the data; the selector may filter further | default `0` (PM) |
| `quote_filter` | `QuoteFilter()`: `min_volume=1`, calendar days `[7, 120]` | the same default filter |

## Source map

| Module | Contents |
|---|---|
| `crashfrontier/data/dataset.py` | `Dataset`, `from_directory`, `from_frames`, `describe` |
| `crashfrontier/data/source.py` | `classify_table`, `open_option_source`, four OptionSource types |
| `crashfrontier/data/schema.py` | `ALIASES`, `ColumnMap`, `canonicalize` |
| `crashfrontier/data/parse.py` | dates, numbers, C/P, settlement, `detect_strike_scale` |
| `crashfrontier/data/forwards.py` / `rates.py` / `spot.py` | auxiliary tables |
| `crashfrontier/data/filters.py` | `QuoteFilter` |
| `crashfrontier/data/cross_section.py` | `CrossSection.from_quotes`, `build_cross_section` |
| `crashfrontier/data/index.py` / `readers.py` | byte-offset date index, table readers |
| `crashfrontier/config.py` | `DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024`, `DEFAULT_LM_RANGE = (-0.35, 0.05)` |
