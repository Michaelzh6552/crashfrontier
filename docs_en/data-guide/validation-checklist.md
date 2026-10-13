# Ingest validation checklist

The first time you switch data (a new OptionMetrics extract, or your own
broker/exchange export), do not jump to `frontier_series`. Work the steps
below. Each has a runnable snippet and a result you should see. On failure,
go back to the matching page; do not invent bid/ask so the later LP
returns a solution.

Real-extract steps read `CRASHFRONTIER_DATA`. Header and unit checks use a
temporary table or a fixed OptionMetrics header string. See
[Market data](../getting-started/data.md).

There is no `Dataset.from_files`. The entry points are `from_directory`,
`Dataset(...)`, `from_frames`, and `from_quotes`. See [index.md](index.md).

## 0. Entry point

| Symptom | Meaning |
|---|---|
| `AttributeError: type object 'Dataset' has no attribute 'from_files'` | that method does not exist |
| `no option tables in directory ...` | you wrote `Dataset("/path/to/market-data")` and the options sit in `RAW_DATA/`. Use `from_directory` |
| `describe()["forwards"] is None` while you have a forward file | constructor scanned a directory; auxiliary tables are not attached |
| `multiple forwards tables` | two forward tables at the same level; pass `forwards=` |
| `in-memory option source cannot be serialized` | `from_frames` `spec()`; a multi-process series falls back to `workers=1` |

## 1. Headers and classification

At least `date, expiry, cp_flag, strike, bid, ask` must map. The
OptionMetrics 38-column header should hit `am_settlement`, not
`am_set_flag`.

```python
from crashfrontier import ColumnMap

header = (
    "secid,date,symbol,symbol_flag,exdate,last_date,cp_flag,strike_price,"
    "best_bid,best_offer,volume,open_interest,impl_volatility,delta,gamma,vega,"
    "theta,optionid,cfadj,am_settlement,contract_size,ss_flag,forward_price,"
    "expiry_indicator,root,suffix,cusip,ticker,sic,index_flag,exchange_d,class,"
    "issue_type,industry_group,issuer,div_convention,exercise_style,am_set_flag"
)
mapping = ColumnMap.detect(header.split(","))
assert mapping.expiry == "exdate"
assert mapping.strike == "strike_price"
assert mapping.bid == "best_bid"
assert mapping.ask == "best_offer"
assert mapping.am_settlement == "am_settlement"
print("headers OK", mapping.rename_map())
```

On your own files, run `ColumnMap.detect(peek_columns(path))` or
`ColumnMap.detect(df.columns)`, and `classify_table(path)` on each table
under `CRASHFRONTIER_DATA` (or `/path/to/market-data`). If a required field
is missing, read the aliases listed in `SchemaError` and use `overrides` or
rename the header; do not edit data rows.

## 2. `describe()`: format, scale, auxiliary sources

```python
import os
from pathlib import Path
from crashfrontier import Dataset

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
info = ds.describe()
print(info["date_format"], info["strike_scale"], info["options"]["kind"])
print(info["forwards"], info["rates"], info["spot"])
```

Check against **your** extract:

- `date_format` matches the file. Ambiguous `07/05/2017` that you want day-first: rebuild with `dayfirst=True`.
- `strike_scale`: OptionMetrics `strike_price` should be 1000; own index points should be 1. `auto` without a reference can misfire; passing the scale is safer.
- `kind`: uncompressed CSV larger than 512 MiB should be `indexed_text`; see [large-files.md](large-files.md).
- `rates=="zero"` when you thought a FRED-style series was attached: classification failed, or you used the constructor.

If you attached a forward table, a two-column rate series, and an index
spot file, expect `forwards=="table"`, `rates=="series"`, and `spot is True`.

## 3. One day’s `expiries()`

Pick a trading day you know, or the last date in the extract.

```python
import os
from pathlib import Path
from crashfrontier import Dataset

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
date = ds.available_dates()[-1]
table = ds.expiries(date)
print(table.to_string(index=False))
assert set(table.columns) == {
    "expiry", "am_settlement", "root", "days", "n_rows",
    "n_put_strikes", "n_call_strikes", "forward", "forward_source",
}
```

Check: `forward` is finite; rows with `days` in 7–120 should not all have
`n_put_strikes` of 0; AM/PM are separate rows. A large `n_rows` with
`n_put_strikes==0` is usually outside the window or a tight quote filter,
not a reason to invent puts. On a typical SPX extract the same Friday can
appear twice (PM/`SPXW` and AM/`SPX`); they are not merged.

## 4. Cross-section: `x`, filter report, source names

```python
import os
from pathlib import Path
from crashfrontier import Dataset

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
cs = ds.cross_section(ds.available_dates()[-1], target_days=30)
s = cs.summary()
print(s)
print(cs.filter_report)
assert 0.8 < float(cs.quotes["x"].median()) < 1.2
assert cs.filter_report["kept"] == s["n_quotes"]
assert cs.filter_report["kept"] > 0
```

`n_quotes`, rate, spot, and the exact expiry depend on the extract. After
a different file, do not insist on a previous kept count; insist on the
median `x` and that the source names match what you attached (`table` /
`series` if those files classified).

Median `x` near 1000: `strike_scale` was not divided enough. Near 0.001:
divided too much. Do not move the threshold `K` to “line up”.

## 5. Rate units

Table value `5.3` on a FRED-style two-column series: default `percent` →
`0.053`. Table already stores `0.053`: `rate_unit="decimal"`. Zero-curve
nodes are always percent. Constant `rates=0.053` is already a decimal. See
[forwards-rates.md](forwards-rates.md).

```python
from crashfrontier.data.rates import open_rate_source
import pandas as pd

s = open_rate_source(pd.DataFrame({"date": ["2021-07-21"], "rf": [5.3]}), unit="percent")
assert abs(s.rate(pd.Timestamp("2021-07-21"), 30) - 0.053) < 1e-12
z = open_rate_source(
    pd.DataFrame({"date": ["2021-07-21", "2021-07-21"], "days": [7, 365], "rate": [5.0, 7.0]}),
    unit="decimal",
)
assert abs(z.rate(pd.Timestamp("2021-07-21"), 7) - 0.05) < 1e-12
print("rate unit OK")
```

## 6. AM/PM

When the same `expiry` has two rows, pin `am_settlement` before computing.
Automatic choice when omitted: more put strikes, tie → PM.
`ambiguous_settlement` True is fine for exploratory notes; it is not a
final rule unless you also report which side was chosen.

## 7. Filter and infeasibility

Compare that day’s raw row count (`option_rows` or `expiries.n_rows`),
each `filter_report` count, and `kept`.

- `kept==0`: first check a column swap (bid/ask reversed, `dayfirst`, strike scaling), then consider a looser quote filter.
- Large `uncrossed`: the market is crossed, or the map is wrong. Do not batch `ask = bid`.
- Solve raises `InfeasibleCrossSectionError`: record the day, check the forward and settlement, change the quote set or the day. Fabricated rows — synthetic bid/ask or Black-76 mids — invalidate the bound.

## 8. Large files

Uncompressed options CSV > 512 MiB:

- `kind=="indexed_text"`
- `{stem}.index.json` (or `{stem}-{hash}.index.json` when stems collide) matches the file’s resolved path, `size`, and `mtime`
- do not `pd.read_csv` the whole table then `from_frames`
- do not raise `memory_limit` above the file size to “skip the index”

A large `.csv.gz` is `chunk_scan`; decompress first.

## 9. One day, then a panel

After the cross-section checks, run one day’s `compute_frontier` with the
**same** `quote_filter` / `dayfirst` / `strike_scale` / `rate_unit`. Look
at `mode` (`hard` is the hard-constraint feasible case), quote count, and
probability/loss scale. Then `term_structure` / `frontier_series`.
`from_frames` + `workers>1` falls back to one process; that is expected.

## 10. Custom-data loop (no vendor extract)

If you have no OptionMetrics, this table proves the entry point works.
Bid/ask are only for a path test.

```python
import pandas as pd
from crashfrontier import ColumnMap, CrossSection, Dataset, QuoteFilter

options = pd.DataFrame(
    {
        "trade_date": ["17/01/2024", "17/01/2024"],
        "maturity": ["16/02/2024", "16/02/2024"],
        "right": ["P", "C"],
        "k": [4300.0, 4800.0],
        "bid_px": [10.0, 11.0],
        "ask_px": [11.0, 12.0],
        "volume": [5, 8],
    }
)
mapping = ColumnMap.detect(
    options.columns,
    overrides={
        "date": "trade_date",
        "expiry": "maturity",
        "cp_flag": "right",
        "strike": "k",
        "bid": "bid_px",
        "ask": "ask_px",
    },
)
ds = Dataset.from_frames(
    options,
    rates=0.053,
    column_map=mapping,
    strike_scale=1.0,
    dayfirst=True,
    quote_filter=QuoteFilter.permissive(),
)
assert ds.available_dates() == [pd.Timestamp("2024-01-17")]
assert ds.describe()["date_format"] == "%d/%m/%Y"
# No forward table, no forward_price, and too few paired strikes: expiries
# forward is NaN and cross_section raises NoEligibleExpiryError. That is
# asking for a real forward, or for from_quotes.
from crashfrontier.errors import NoEligibleExpiryError

try:
    ds.cross_section("17/01/2024", days=30)
    raise SystemExit("without a forward this panel should not be eligible")
except NoEligibleExpiryError:
    pass

cs2 = CrossSection.from_quotes(
    options,
    date="17/01/2024",
    expiry="16/02/2024",
    forward=4790.5,
    rate=0.053,
    column_map=mapping,
    strike_scale=1.0,
    dayfirst=True,
    quote_filter=QuoteFilter.permissive(),
)
assert cs2.forward_source == "user"
assert 0.8 < float(cs2.quotes["x"].median()) < 1.2
print("custom path OK", cs2.summary())
```

`17/01/2024` is 17 January with `dayfirst` left unset, because 17 is not a
month. `dayfirst=True` in the listing selects the same day. `from_quotes`
`x` is near 1. If `from_frames` fails for lack of a forward, add a real
forward table or use `from_quotes(..., forward=F)`. Do not duplicate the
put/call pair to fill in parity.

## Page index

| Check | Page |
|---|---|
| Directory vs constructor | [from-directory.md](from-directory.md) |
| In-memory multi-day tables | [from-frames.md](from-frames.md) |
| One day, one expiry | [from-quotes.md](from-quotes.md) |
| Suffixes and aliases | [formats-columns.md](formats-columns.md) |
| Dates / scale / settlement | [dates-strikes.md](dates-strikes.md) |
| Forward chain and rates | [forwards-rates.md](forwards-rates.md) |
| `filter_report` | [filters.md](filters.md) |
| Large-file date index | [large-files.md](large-files.md) |
