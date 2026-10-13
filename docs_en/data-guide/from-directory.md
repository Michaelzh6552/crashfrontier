# `Dataset.from_directory` and the `Dataset(...)` constructor

Use `from_directory` when a directory holds options, forwards, rates, and
spot together. When you already know each path, pass `options`, `forwards`,
`rates`, and `spot` to the constructor.

Point `root` at a folder you already hold. Real-extract scripts should set
`CRASHFRONTIER_DATA`. See [Market data](../getting-started/data.md).

## Signature

```python
Dataset.from_directory(root, **kwargs) -> Dataset

Dataset(
    options,
    *,
    forwards=None,
    rates=None,
    spot=None,
    column_map=None,          # ColumnMap, or dict (treated as detect overrides)
    strike_scale="auto",
    dayfirst=None,
    cp_values=None,
    rate_unit="percent",
    forward_sources=("table", "column", "parity"),
    quote_filter=QuoteFilter(),
    cache_dir=None,
    memory_limit=512 * 1024 * 1024,
)
```

`from_directory` buckets files by header, then calls the same constructor.
`kwargs` are forwarded as-is. Explicit `forwards` / `rates` / `spot`
**override** the auto-chosen files (passing `None` means drop that auxiliary
table).

## Which files `from_directory` sees

Implementation: `_list_directory_tables`.

1. `root` must be an existing directory, else `SchemaError: ... is not a directory`.
2. Only `root/`, plus `root/RAW_DATA/` if it exists.
3. **One level each, no recursion.** `RAW_DATA/foo/bar.csv` is not seen.
4. Regular files only. Subdirectories and non-tables such as `.DS_Store`: skip classification when `file_kind == "unknown"`.
5. `classify_table(path)` reads the header only (`peek_columns`), not the body.
6. Unclassified files log info `ignoring unrecognized table <name>` and are ignored.
7. No option table: `SchemaError: no option tables in directory ...`.

Classification rules (first match wins):
[formats-columns.md](formats-columns.md#directory-classification-classify_table).
Summary:

| Header condition | Kind |
|---|---|
| strike, C/P, and bid aliases all present | `options` (**all** such files are used) |
| forward alias and expiry alias, and no strike alias | `forwards` |
| `days` column and a rate alias | `zero_curve` |
| exactly two columns: one is a date alias, the other starts with `dgs` or is in `{rate, yield, tbill, riskfree, rf, value}` | `rate_series` |
| date alias and a spot-price alias (`close` / `price` / `adj_close` / `spindx` / `level` / `last` / `value`, **not** the column name `spot`) | `spot` |
| else | `unknown` |

Conflicts within a kind:

- Several `forwards`: `SchemaError: multiple forwards tables (...); pass forwards= explicitly`.
- Both `zero_curve` and `rate_series`, or several tables of the same rate kind: `multiple rate tables; pass rates= explicitly`.
- Every spot candidate whose filename contains `vix` or `vx` is dropped, including when it is the only spot file. If none remain, the dataset has no spot table.
- If more than one spot file remains, prefer names containing `spx` / `spot` / `secprd` / `underlying` / `index`. If more than one still remains, `SchemaError: multiple spot tables`.
- Several `options`: all are opened. Canonical column name sets must match, else `option file columns do not match`.

A VIX history file often classifies as `spot` from its header (`CLOSE`) and is then dropped by that filename rule. `test_from_directory_excludes_vix` covers the exclusion.

## Constructor: Dataset is not from_directory {#constructor-dataset-is-not-from_directory}

`options` may be a single path, a list of paths, a directory that **contains
only option tables**, or a DataFrame.

If `options` is a directory, `_as_option_paths` / `_directory_option_files`
look **only at that level** for `classify_table == "options"`:

- They do **not** enter `RAW_DATA/`.
- They do **not** attach forwards, rates, or spot. A sibling `forward.csv` is ignored unless you pass `forwards=`.
- Against a root whose options live only in `RAW_DATA/`: `SchemaError: no option tables in directory ...`.
- Against the options folder itself: the option file opens, but
  `describe()["forwards"] is None`, `rates == "zero"`, `spot is False`.
  Without a forward reference price, `strike_scale="auto"` guesses `1000`
  from a median strike `> 100000` and warns.

Path list: `Dataset([opt_a, opt_b], forwards=..., rates=...)`. Canonical
column name sets must match.

`column_map`:

- `None`: `ColumnMap.detect` on `peek_columns` of the first option table.
- `ColumnMap` object: used as-is; aliases are not filled in.
- `dict`: passed as `ColumnMap.detect(..., overrides=dict)`. Fields you do not name still fill from aliases.

`CrossSection.from_quotes` **does not** accept a dict; see
[from-quotes.md](from-quotes.md).

## Your extract (`CRASHFRONTIER_DATA`)

```python
import os
from pathlib import Path
from crashfrontier import Dataset
from crashfrontier.errors import SchemaError

root = Path(os.environ["CRASHFRONTIER_DATA"])
assert root.is_dir()
assert not hasattr(Dataset, "from_files")

ds = Dataset.from_directory(root)
info = ds.describe()
print(info["options"]["kind"], info["options"]["n_dates"])
print(info["date_format"], info["strike_scale"])
print(info["forwards"], info["rates"], info["spot"])
print(info["quote_filter"]["min_volume"], Path(info["cache_dir"]).name)

dates = ds.available_dates()
assert dates[0] == info["options"]["first_date"]
assert dates[-1] == info["options"]["last_date"]

# pick a trading day you know, or the last date in the extract
table = ds.expiries(dates[-1])
print(table.to_string(index=False))

cs = ds.cross_section(dates[-1], target_days=30)
print(cs.summary())
assert 0.8 < float(cs.quotes["x"].median()) < 1.2

# the constructor does not stand in for from_directory
try:
    Dataset(root)
except SchemaError as exc:
    print(exc)  # no option tables ... if options sit only in RAW_DATA/
```

Date range, `kind`, and which auxiliaries attach depend on **your** files.
OptionMetrics IvyDB extracts often have `"%Y-%m-%d"` dates, `strike_scale`
1000, a forward table, and a FRED-style rate series. An uncompressed
options CSV larger than the default 512 MiB uses the byte-offset date
index (`indexed_text`). The first open with no matching cache scans the
whole text (progress bar over the file); runtime depends on disk and file
size. When the index already matches `size`/`mtime`, it is reused. Details:
[large-files.md](large-files.md).

`cross_section` requires `expiry`, `days`, or `target_days`; omitting all three
raises `ValueError: cross_section requires expiry, days, or target_days`.

On a typical SPX extract, the same Friday can appear twice in `expiries()`
(PM/`SPXW` and AM/`SPX`); they are not merged. Pin `am_settlement` when
both hit.

## Mini-directory (no vendor extract)

Shows classification, `RAW_DATA/`, and that the constructor does not attach
auxiliary tables. Same layout as a pytest `tmp_path` fixture.

```python
from pathlib import Path
import tempfile
import pandas as pd
from crashfrontier import Dataset
from crashfrontier.data import classify_table
from crashfrontier.errors import SchemaError

def write_mini(root: Path) -> Path:
    raw = root / "RAW_DATA"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "exdate": ["2021-08-04", "2021-08-04"],
            "cp_flag": ["P", "C"],
            "strike_price": [3900000, 4100000],
            "best_bid": [12.0, 9.0],
            "best_offer": [12.4, 9.3],
            "volume": [10, 8],
            "am_settlement": [0, 0],
            "symbol": ["SPXW 210804P3900000", "SPXW 210804C4100000"],
        }
    ).to_csv(raw / "option_price.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2021-07-21"],
            "expiration": ["2021-08-04"],
            "AMSettlement": [0],
            "ForwardPrice": [4000.0],
        }
    ).to_csv(raw / "forward.csv", index=False)
    pd.DataFrame(
        {"observation_date": ["2021-07-21"], "DGS1MO": [5.0]}
    ).to_csv(raw / "DGS1MO.csv", index=False)
    pd.DataFrame({"Date": ["2021-07-21"], "Price": [3999.0]}).to_csv(
        raw / "SPX.csv", index=False
    )
    pd.DataFrame({"DATE": ["07/21/2021"], "CLOSE": [17.01]}).to_csv(
        raw / "VIX_History.csv", index=False
    )
    (raw / "notes.txt").write_text("a,b\n1,2\n", encoding="utf-8")
    return root

with tempfile.TemporaryDirectory() as tmp:
    root = write_mini(Path(tmp) / "market-data")
    raw = root / "RAW_DATA"
    assert classify_table(raw / "option_price.csv") == "options"
    assert classify_table(raw / "forward.csv") == "forwards"
    assert classify_table(raw / "DGS1MO.csv") == "rate_series"
    assert classify_table(raw / "SPX.csv") == "spot"
    assert classify_table(raw / "VIX_History.csv") == "spot"
    assert classify_table(raw / "notes.txt") == "unknown"

    ds = Dataset.from_directory(root, cache_dir=root / "CACHE")
    info = ds.describe()
    assert info["forwards"] == "table"
    assert info["rates"] == "series"
    assert info["spot"] is True
    assert info["strike_scale"] == 1000.0
    assert info["date_format"] == "%Y-%m-%d"
    cs = ds.cross_section("2021-07-21", days=14)
    assert cs.forward == 4000.0
    assert cs.forward_source == "table"
    assert cs.rate == 0.05  # 5.0 percent -> 0.05
    assert cs.rate_source == "series"
    assert cs.spot == 3999.0
    assert cs.quotes["strike"].tolist() == [3900.0, 4100.0]
    print("mini from_directory", cs.summary())

    try:
        Dataset(root, cache_dir=root / "CACHE2")
        raise SystemExit("Dataset(root) should not see RAW_DATA/")
    except SchemaError as exc:
        assert "no option tables" in str(exc)

    # same-level directory: constructor takes options only; pass auxiliaries explicitly
    only_opt = Dataset(raw, cache_dir=root / "CACHE3")
    bare = only_opt.describe()
    assert bare["forwards"] is None
    assert bare["rates"] == "zero"
    assert bare["spot"] is False
    print("constructor-on-options-dir auxiliaries", bare["forwards"], bare["rates"], bare["spot"])
```

Five `classify_table` results match the table above; the `from_directory`
cross-section has forward 4000, rate 0.05, spot 3999, strikes already
divided by 1000; `Dataset(root)` fails; `Dataset` on the options folder
opens quotes with empty auxiliary sources.

Copy `forward.csv` and call `from_directory` again:
`SchemaError: multiple forwards` (`test_from_directory_duplicate_forwards`).

## `describe()` / `expiries()` / `spec()`

`describe()` keys (`test_describe_fields`): `options`, `date_format`,
`strike_scale`, `forwards`, `rates`, `spot`, `quote_filter`, `cache_dir`.

`options` subkeys: `files`, `kind`, `n_dates`, `first_date`, `last_date`,
`columns` (source name → canonical name).

`kind`: `indexed_text` | `frame` | `parquet` | `chunk_scan`. A tiny CSV is
usually `frame` (total size ≤ `memory_limit`). An uncompressed CSV larger
than `memory_limit` is `indexed_text`.

`expiries(date)` is one row per `(expiry, am_settlement, root)`. `n_rows`
is the raw count before filtering; `n_put_strikes` is the number of put
strikes **after `QuoteFilter.apply_rows`** whose log-moneyness lies in the
default `(-0.35, 0.05)`. So a row with `days < 7` or `> 120` can have a
large `n_rows` and still `n_put_strikes` of 0. If no forward is found, that
row’s `forward` is NaN and `forward_source="none"`; `cross_section` will
not select it.

`spec()` serializes **file-backed** option sources only, for
`frontier_series(..., workers>1)`. An in-memory option source
(`from_frames`, or a small file read into `FrameSource` because a DataFrame
was passed at construction) raises
`ValueError: in-memory option source cannot be serialized`. A small CSV
opened by path and then loaded into `FrameSource` can still write a path
list from `spec()`, as long as `options` was not a DataFrame at
construction.

## Selecting a cross-section

Candidates: finite `forward`, and `min_days ≤ days ≤ max_days` (from
`quote_filter`, default 7–120). Then:

1. `expiry=`: exact expiry (same `dayfirst` rules).
2. Else `days=`: exact calendar day count.
3. Else `target_days=`: inside `window` (default = the filter’s day bounds), with `n_put_strikes >= min_put_strikes`, minimize `|days - target_days|`; ties take the earlier expiry.
4. Several AM/PM and no `am_settlement`: `ambiguous_settlement=True`, a warning, sort by `n_put_strikes` descending then `am_settlement` ascending (**tie → PM=0 first**).
5. Empty set: `NoEligibleExpiryError`; the message includes the full `expiries` table.
6. Observation date missing: `NoSuchDateError` (`rows` also names the neighboring dates).

`nearest_date(date, direction="backward"|"forward")` snaps to a trading day
that has options. A bad `direction` raises `ValueError`.

`cross_sections(date)` returns every eligible
`(expiry, settlement, root)` that day. AM/PM are not merged.

User override: `cross_section(..., forward=F, rate=r)` sets the sources to
`"user"` (`test_user_forward_and_rate`).

## Common mistakes

- `Dataset.from_files(...)`: does not exist.
- `Dataset("/path/to/market-data")` instead of `from_directory` when options live in `RAW_DATA/`.
- Passing the data root to CLI `--options`: that is not directory classification. In Python use `from_directory` or the constructor plus explicit paths.
- Options split across files with mismatched headers: `SchemaError`.
- Two forward tables at the same level: pass `forwards=` to name one.
