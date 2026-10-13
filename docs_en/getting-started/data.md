# Market data

After [install](install.md), point the library at tables from a licensed
vendor extract or from files you already maintain. The package ships as code
only — nothing in the install step downloads option quotes, forwards,
interest rates, or any other market series.

## OptionMetrics IvyDB US

The examples in this documentation were built on **OptionMetrics IvyDB US** extracts. IvyDB is licensed data; obtain an extract through your institution — typically a WRDS subscription or a direct OptionMetrics license.

IvyDB column names such as `exdate`, `strike_price`, `best_bid`, and `best_offer` are recognized automatically. Raw IvyDB `strike_price` values are usually index points times 1,000; `Dataset` infers that scale by default, while `CrossSection.from_quotes` defaults to `strike_scale=1.0` and needs `strike_scale=1000` if you pass those raw strikes.

## Your own tables

Any quote table is fine if the required option fields can be mapped. Headers do not have to match IvyDB. Common aliases are listed in [Formats and column aliases](../data-guide/formats-columns.md). If a name is not in that list, pass `ColumnMap.detect(columns, overrides=...)` — do not construct a partial `ColumnMap(...)` that omits required fields.

## Required and recommended option columns

These six fields are required. Missing one raises `SchemaError` and lists the aliases the reader tried.

| Field | Role | Common source names |
|---|---|---|
| `date` | Observation / trade date | `date`, `trade_date`, `observation_date` |
| `expiry` | Option expiration | `exdate`, `expiration`, `expiry`, `maturity` |
| `cp_flag` | Call or put (`C` / `P`) | `cp_flag`, `option_type`, `call_put` |
| `strike` | Strike, later divided by `strike_scale` | `strike`, `strike_price` |
| `bid` | Bid in the same units as the underlying | `bid`, `best_bid` |
| `ask` | Ask in the same units as the underlying | `ask`, `best_offer`, `best_ask` |

These four are optional but useful:

| Field | Why include it |
|---|---|
| `volume` | The default quote filter keeps `volume >= 1`. If the column is absent or entirely missing, that rule is skipped rather than dropping every row. |
| `open_interest` | Available for a tighter filter if you set `min_open_interest`. |
| `impl_volatility` | Carried onto the cross-section for inspection; not required to solve. |
| `forward_price` | Used when there is no separate forward table. |

`symbol`, `root`, `optionid` / `quote_id`, and `am_settlement` are also recognized when present. AM and PM settlements are never merged.

Default `QuoteFilter` also requires calendar days in `[7, 120]`. Turn the volume floor off with `QuoteFilter.permissive()`. See [Filters](../data-guide/filters.md).

## Forwards, rates, and spot

A cross-section needs a **finite forward**. `Dataset` tries, in order: a dedicated forward table, a `forward_price` column on the option rows, then put–call parity. Parity needs a rate and enough paired call/put mids; it is a fallback, not a reason to edit bid and ask. You can also pass `forward=` when you already know the value.

Rates set the discount \(D=\exp(-r\cdot\mathrm{days}/365)\). Choose one of:

- a two-column date/value series (default `rate_unit="percent"`, so `5.3` becomes `0.053`; use `rate_unit="decimal"` if the file already stores decimals)
- a zero curve with a `days` column (always read as percent)
- a Python float, already a decimal, e.g. `rates=0.05`
- omit the series: \(r=0\)

Spot is optional. It is a reference for automatic strike scaling and for parity, and it is stored on the cross-section. A filename containing `vix` or `vx` is never attached as spot. Directory classification also ignores a price column named only `spot`; it looks for `close`, `price`, `adj_close`, `spindx`, `level`, `last`, or `value`.

Details: [Forwards and rates](../data-guide/forwards-rates.md).

## Directory layout

`Dataset.from_directory` looks at the folder you pass and, if it exists, a `RAW_DATA/` directory immediately inside it. Each of those two levels is scanned once; nested folders are not.

Classification uses **headers**, not filenames. The names below are examples only.

```text
/path/to/market-data/
    options.csv       # strike + C/P + bid  → options
    forwards.csv      # forward + expiry, no strike  → forwards
    rates.csv         # date + rate, or a days curve  → rates
    spot.csv          # date + close / price / …  → spot
```

An OptionMetrics-style layout with the tables under `RAW_DATA/` is also fine. Extra files whose headers do not match a known kind are ignored. Two forward tables, or two rate tables, require you to pass `forwards=` or `rates=` explicitly.

Prefer `from_directory` when options and auxiliaries sit together. `Dataset("/path/to/market-data")` looks for option tables in that folder only: it does not enter `RAW_DATA/`, and it does not attach forward, rate, or spot files.

Full rules: [Read from a directory](../data-guide/from-directory.md).

### The date index is not market data

Large uncompressed option CSVs (default threshold 512 MiB) are not loaded in one piece. On the first open the library writes a byte-range date index under `cache_dir` and reuses it on later opens. That JSON file is a local index the reader generates from your own files.

If you omit `cache_dir`, the default is a `CACHE/` folder two levels above the first option file. For a flat directory like the tree above, pass `cache_dir` so the index stays next to your files. The index is rebuilt when the source file’s size or modification time changes.

See [Large-file index](../data-guide/large-files.md).

## Starting point A: a directory you already have

Replace the path with the folder that holds your licensed extract or your own tables.

```python
from crashfrontier import Dataset

ds = Dataset.from_directory(
    "/path/to/market-data",
    cache_dir="/path/to/market-data/CACHE",
)
info = ds.describe()
print(info["date_format"], info["strike_scale"])
print(info["forwards"], info["rates"], info["spot"])
print(info["options"]["n_dates"], info["options"]["files"])

dates = ds.available_dates()
print(dates[0].date(), dates[-1].date())

cs = ds.cross_section(dates[-1], target_days=30)
print(cs.summary())
print(cs.filter_report)
```

`cross_section` requires one of `expiry`, `days`, or `target_days`. Inspect `ds.expiries(date)` before pinning a contract. `describe()` reports the column map, strike scale, and which forward and rate sources were attached.

## Starting point B: a DataFrame

No directory is required. Use this when quotes are already in memory.

**Several observation dates.** `Dataset.from_frames` takes an options frame and optional forward, rate, and spot frames:

```python
from crashfrontier import Dataset

ds = Dataset.from_frames(options, forwards=forwards, rates=rates, spot=spot)
cs = ds.cross_section(observation_date, target_days=30)
```

**One date and one expiry.** `CrossSection.from_quotes` needs the quotes, that date, that expiry, and a forward. Rate and spot are optional (`rate` defaults to `0.0`). `strike_scale` defaults to `1.0` (no auto-detect). `column_map`, if you pass one, must be a `ColumnMap`, not a dict.

```python
from crashfrontier import CrossSection

cs = CrossSection.from_quotes(
    quotes,
    date=trade_date,
    expiry=expiry,
    forward=forward,
    rate=rate,
)
print(cs.summary())
```

If the frame has no date or expiry columns, the `date` and `expiry` arguments are written onto the table before mapping.

For a complete `from_quotes` listing, including column mapping and `strike_scale`, see [One day, one expiry](../data-guide/from-quotes.md). Multi-day frames: [Read from in-memory tables](../data-guide/from-frames.md).

## Next

- [Data guide](../data-guide/index.md) — formats, dates, filters, and the validation checklist
- [Select a cross-section](../user-guide/select-cross-section.md) — `expiries`, `nearest_date`, AM/PM
- [First crash frontier](first-frontier.md) — compute and plot once you have a `CrossSection`
