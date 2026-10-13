# File formats and column aliases

`crashfrontier.data.readers` reads tables. `ColumnMap.detect` maps columns.
`classify_table` buckets a directory by header only.

## Suffix → `file_kind`

The last suffix **after compression suffixes are stripped** (`test_file_kind`).

| kind | Suffixes |
|---|---|
| `text` | `.csv` `.txt` `.tsv` `.dat` (case-insensitive) |
| `text_gz` | those plus `.gz` or `.zip`, e.g. `quotes.csv.gz`, `quotes.txt.zip` |
| `parquet` | `.parquet` (`quotes.parquet.gz` is still parquet: `.gz` is stripped first) |
| `sas` | `.sas7bdat` |
| `stata` | `.dta` |
| `excel` | `.xls` `.xlsx` `.xlsm` `.xlsb` |
| `unknown` | everything else, including bare `.gz`, `.md` |

`unknown` cannot `peek_columns` / `read_table`
(`SchemaError: unsupported table file type`). `from_directory` skips these
files.

Text reads use `encoding="utf-8-sig"`, `compression="infer"`,
`low_memory=False`. `sniff_delimiter` inspects the first 64 KiB among `,;`,
tab, and `|`; failure or an empty file → `,`. BOM + quoted headers keep the
original names (`test_bom_header_names`): `Date` / `Price`.

Missing PyArrow raises `ImportError` and mentions
`pip install 'crashfrontier[parquet]'`. The `excel` extra installs
`openpyxl` for `.xlsx` / `.xlsm`, `xlrd` for `.xls`, and `pyxlsb` for
`.xlsb`. SAS uses `format="sas7bdat"`, `encoding="latin-1"`. Excel is read
whole (`iter_table` yields once). Integer columns are cast to `float64` so
whole-number floats retain the canonical numeric dtype.

Backend choice for large files (`IndexedTextSource` / `ChunkScanSource` /
`ParquetSource`): [large-files.md](large-files.md).

## `normalize_name`

Before names are compared: strip a UTF-8 BOM, unwrap paired quotes, strip,
lowercase, and replace space / `-` / `.` with `_`.

```python
from crashfrontier.data.schema import normalize_name

assert normalize_name("\ufeffDate") == "date"
assert normalize_name('"Best Bid"') == "best_bid"
assert normalize_name("AM-Set.Flag") == "am_set_flag"
```

`ColumnMap.detect` stores the **original spelling** (`DATE` stays `DATE`).
CSV I/O uses that source name.

## Option aliases (`ALIASES`, in written order)

Short names `type` / `flag` / `id` / `k` are used only when no longer alias
hits.

| Canonical name | Aliases (long then short) |
|---|---|
| `date` | `date` `quote_date` `quotedate` `trade_date` `tradedate` `obs_date` `observation_date` `as_of_date` `asofdate` `datadate` |
| `expiry` | `exdate` `expiration` `expiry` `expiration_date` `expirationdate` `exp_date` `expdate` `maturity` `maturity_date` |
| `cp_flag` | `cp_flag` `cpflag` `call_put` `callput` `put_call` `putcall` `option_type` `optiontype` `type` `cp` `right` `flag` |
| `strike` | `strike_price` `strikeprice` `strike` `strike_px` `k` |
| `bid` | `best_bid` `bestbid` `bid` `bid_price` `bidprice` `bid_px` |
| `ask` | `best_offer` `bestoffer` `best_ask` `bestask` `ask` `offer` `ask_price` `askprice` `ask_px` |
| `volume` | `volume` `vol` `trading_volume` `tradingvolume` `total_volume` `totalvolume` |
| `open_interest` | `open_interest` `openinterest` `oi` `open_int` |
| `impl_volatility` | `impl_volatility` `implvolatility` `implied_volatility` `impliedvolatility` `impl_vol` `implvol` `iv` |
| `quote_id` | `optionid` `option_id` `contract_id` `contractid` `id` |
| `am_settlement` | `am_settlement` `amsettlement` `am_set_flag` `amsetflag` `am_flag` `settlement` |
| `symbol` | `symbol` `option_symbol` `optionsymbol` `contract_symbol` `contract` `ticker_symbol` |
| `root` | `root` `root_symbol` `rootsymbol` `underlying` `underlying_symbol` |
| `forward_price` | `forward_price` `forwardprice` `forward` `fwd` `fwd_price` |
| `expiry_indicator` | `expiry_indicator` `expiryindicator` |

Required: `REQUIRED_FIELDS = ("date", "expiry", "cp_flag", "strike", "bid", "ask")`.
Missing: `SchemaError: missing required field 'expiry'; aliases [...]; columns [...]`.

One source column cannot map to two fields. `overrides={"expiry": "date"}`
with no second date column treats `expiry` as missing
(`test_detect_conflict_treats_later_field_as_missing`). Unknown override key:
`unknown override field`. Override pointing at a missing column:
`override column ... is not among`.

## OptionMetrics 38-column header

A common IvyDB-style export (`OPTIONMETRICS_HEADER` in `tests/test_schema.py`)
has 38 columns. When both `am_settlement` and `am_set_flag` exist,
`ColumnMap.detect` picks **`am_settlement`**; `quote_id="optionid"`;
`forward_price="forward_price"`.

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
assert mapping.am_settlement == "am_settlement"
assert mapping.am_settlement != "am_set_flag"
assert mapping.quote_id == "optionid"
assert mapping.expiry == "exdate"
assert mapping.strike == "strike_price"
assert mapping.bid == "best_bid"
assert mapping.ask == "best_offer"
assert mapping.rename_map()["exdate"] == "expiry"
print(mapping.usecols()[:8])
```

`ColumnMap.optionmetrics()` writes that same set of standard source names
(lowercase). If the header is `DATE` / `EXDATE`, do not read with
`optionmetrics()`; use `detect` so the uppercase source names are kept
(`test_detect_uppercase_names_preserve_original_spelling`, variant
`uppercase_semicolon_mdy`).

## `detect` vs a hand-built `ColumnMap`

```python
# correct: unnamed fields still fill from aliases
ColumnMap.detect(columns, overrides={"date": "MyDate", "strike": "k"})

# wrong: the constructor requires date/expiry/cp_flag/strike/bid/ask all at once,
# and optional fields are not auto-matched
# ColumnMap(date="MyDate", strike="k")
```

`Dataset`’s `column_map=dict` takes the first path. `CrossSection.from_quotes`
must receive a `ColumnMap` object.

## Extra `canonicalize` rules

- Numeric columns: `parse_numeric` strips `,` and `$`; empty string, `.`, `nan` → NaN.
- `strike /= strike_scale`.
- `quote_id`: integer-valued floats become integer strings (`83164582.0` → `"83164582"`); missing id → `r{position}`.
- `root`: if `symbol` exists, the token before the first space (`SPX 210804C...` → `SPX`); else the `root` column; else `""`.
- `am_settlement`: see [dates-strikes.md](dates-strikes.md#am-pm-settlement).
- Drop rows whose `date` / `expiry` / `strike` / `bid` / `ask` / `cp_flag` is missing.

## Directory classification: `classify_table` {#directory-classification-classify_table}

Normalized header only, fixed order (`source.py`):

1. **options**: a strike alias **and** a cp_flag alias **and** a bid alias. The header need not already contain date/expiry (those fail later at `detect`).
2. **forwards**: a forward alias in `{forwardprice, forward_price, forward, fwd, fwd_price, f}` **and** an expiry alias (the option `expiry` set) **and no** strike alias.
3. **zero_curve**: a `days` column and a hit in `{rate, zero_rate, yield, zero}`.
4. **rate_series**: exactly two columns; one is a date alias; the other, after normalization, starts with `dgs` or is in `{rate, yield, tbill, riskfree, rf, value}`.
5. **spot**: a date alias **and** `{close, price, adj_close, adjclose, spindx, level, last, value}`.
6. Else `unknown`.

Difference from `SpotTable`: **reading** spot accepts one extra price alias,
`spot`; **classification** does not. A two-column `date,spot` file is
`unknown` to `classify_table` and is ignored by `from_directory`. Pass
`spot=` to the constructor to use it.

Typical files after classification (headers, not filenames; none of these
ship with the package):

| Example file | Classification |
|---|---|
| `option_price.csv` (strike + C/P + bid) | `options` |
| `forward.csv` (forward + expiry, no strike) | `forwards` |
| FRED `DGS1MO.csv` (second column `dgs1mo`) | `rate_series` |
| index levels `SPX.csv` (`Price`) | `spot` |
| `VIX_History.csv` (`CLOSE`) | `spot`, then dropped by the `vix` / `vx` filename rule even if it is the only spot file |
| `vx_contract_daily_*.csv` (`Trade Date` + `Close`) | `spot`, then dropped by the same filename rule |
| `notes.txt` with two alias-free columns | `unknown` |

Options beat “a rate table that happens to have `days`”, so an option file
that contains `days` is still options.

## Semicolon headers and MDY dates

```python
from pathlib import Path
import tempfile
import pandas as pd
from crashfrontier import Dataset
from crashfrontier.data.readers import file_kind, peek_columns, sniff_delimiter
from crashfrontier.data.schema import ColumnMap

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "OPTIONS.txt"
    path.write_text(
        "DATE;EXDATE;CP_FLAG;STRIKE_PRICE;BEST_BID;BEST_OFFER;VOLUME\n"
        "07/21/2021;08/04/2021;P;4000000;10;10.2;4\n",
        encoding="utf-8",
    )
    assert file_kind(path) == "text"
    assert sniff_delimiter(path) == ";"
    cols = peek_columns(path)
    assert cols[0] == "DATE"
    mapping = ColumnMap.detect(cols)
    assert mapping.date == "DATE"
    assert mapping.expiry == "EXDATE"
    ds = Dataset(
        path,
        rates=0.05,
        column_map=mapping,
        strike_scale=1000.0,
        dayfirst=False,
    )
    rows = ds.option_rows("2021-07-21")
    assert rows["strike"].tolist() == [4000.0]
    assert rows["date"].iloc[0] == pd.Timestamp("2021-07-21")
    print(rows[["date", "expiry", "strike", "bid", "ask"]])
```

Semicolon, uppercase header, `07/21/2021` parsed month-day-year; strike
divided by 1000. Reduced form of the test variant `uppercase_semicolon_mdy`.
