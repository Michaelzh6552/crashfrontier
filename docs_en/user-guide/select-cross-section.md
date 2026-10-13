# Select the option cross-section

A `CrossSection` is one observation date, one expiry, and one settlement. The script lists available expiries, selects with `expiry`, `days`, or `target_days`, and handles days that carry both AM and PM.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `select_cross_section.py` and run `python select_cross_section.py`.

```python
import os
from pathlib import Path

from crashfrontier import CrossSection, Dataset, QuoteFilter
from crashfrontier.errors import NoEligibleExpiryError, NoSuchDateError


def main() -> None:
    DATA_ROOT = Path(os.environ["CRASHFRONTIER_DATA"])
    ds = Dataset.from_directory(DATA_ROOT)

    print("nearest trading day: 2017-07-04 ->", ds.nearest_date("2017-07-04"))
    try:
        ds.option_rows("2017-07-04")
    except NoSuchDateError as exc:
        print("NoSuchDateError:", exc)

    expiries = ds.expiries("2017-07-05")
    print("\n2017-07-05 expiry table (7–21 days):")
    print(expiries.loc[expiries["days"].between(7, 21)].to_string(index=False))

    by_target = ds.cross_section("2017-07-05", target_days=14, window=(7, 21))
    by_days = ds.cross_section("2017-07-05", days=14)
    by_expiry = ds.cross_section("2017-07-05", expiry="2017-07-19")
    print("\nsame cross-section under three selectors:",
          by_target.expiry == by_days.expiry == by_expiry.expiry,
          "am_settlement", by_target.am_settlement)

    print("\n14-day cross-section:")
    print(by_target.summary())
    print("filter_report", by_target.filter_report)

    pm = ds.cross_section("2017-07-05", expiry="2017-07-21", am_settlement=0)
    am = ds.cross_section("2017-07-05", expiry="2017-07-21", am_settlement=1)
    print("\nsame calendar expiry, different settlement:")
    print("PM", pm.root, "am_settlement", pm.am_settlement, "n_quotes", len(pm.quotes), "F", pm.forward)
    print("AM", am.root, "am_settlement", am.am_settlement, "n_quotes", len(am.quotes), "F", am.forward)

    auto = ds.cross_section("2017-07-05", expiry="2017-07-21")
    print(
        "unspecified settlement: am_settlement=",
        auto.am_settlement,
        "root=",
        auto.root,
        "ambiguous=",
        auto.ambiguous_settlement,
    )

    try:
        ds.cross_section("2017-07-05", days=2)
    except NoEligibleExpiryError as exc:
        print("\ndays=2 outside default [7, 120] window:", str(exc).splitlines()[0])

    loose = Dataset.from_directory(DATA_ROOT, quote_filter=QuoteFilter.permissive())
    loose_cs = loose.cross_section("2017-07-05", days=14)
    print("\ndefault min_volume=1 kept", by_target.filter_report["kept"],
          "; permissive kept", loose_cs.filter_report["kept"])

    # Rebuild a from_quotes sheet from the selected cross-section (API demo, not new data)
    sheet = by_target.quotes[["strike", "cp_flag", "bid", "ask", "volume"]].copy()
    rebuilt = CrossSection.from_quotes(
        sheet,
        date="2017-07-05",
        expiry="2017-07-19",
        forward=float(by_target.forward),
        rate=float(by_target.rate),
        spot=float(by_target.spot),
        am_settlement=0,
        strike_scale=1.0,
    )
    print("from_quotes n_quotes", rebuilt.summary()["n_quotes"],
          "forward_source", rebuilt.forward_source)


if __name__ == "__main__":
    main()
```

## Selection rules

### `expiries` is a diagnostic table

`ds.expiries(date)` lists candidate expiries. It does not compute a frontier. Each `(expiry, am_settlement, root)` is one row with `days`, filtered put/call strike counts, the forward, and its source. When the entire forward chain fails, that row has `forward` as NaN, `forward_source="none"`, and `n_put_strikes=0`. `expiries()` swallows `MissingForwardError`, so the latter often never appears on screen.

A candidate must have a finite forward and a calendar day inside the filter window (default 7–120). On 2017-07-05 the 0-, 2-, and 5-day contracts have `n_put_strikes=0` because the day window dropped them, even though quotes exist.

### Selector precedence

```python
ds.cross_section(date, expiry=...)
ds.cross_section(date, days=...)
ds.cross_section(date, target_days=..., window=(lo, hi))
```

Omitting all three raises `ValueError: cross_section requires expiry, days, or target_days`. When all three are passed, the code takes the first of `expiry` → `days` → `target_days` in an `if/elif` chain and ignores the rest. `window` has an effect only with `target_days`; if `window` is omitted, the filter's `min_days` / `max_days` are used. Ties take the earlier expiry.

`target_days` also requires `n_put_strikes >= min_put_strikes` (default 0).

`cross_section(..., forward=F)` **cannot** turn an expiry that has no finite forward back into a candidate. It only overwrites the forward value on the row that was already selected, and marks `forward_source` as `user`.

### AM / PM are never merged

`am_settlement=1` is morning settlement; `0` is afternoon. When there is no settlement column: `root` exactly `"SPX"` → AM, everything else (including `SPXW`) → PM. On 2017-07-05 the 14-day expiry is PM only; the same day's 16-day expiry has both an SPXW afternoon session and an SPX morning session.

If `am_settlement` is omitted and both settlements match, a warning is issued and the expiry with more put strikes is preferred, PM on a tie. `expiry="2017-07-21"` selects AM (88 put strikes vs 58) and sets `ambiguous_settlement=True`. Record `cs.am_settlement` and `cs.root`.

`ds.cross_sections(date)` returns every eligible `(expiry, settlement, root)` cross-section that day.

### Filtering happens before expiry selection

Default `QuoteFilter()`: `volume >= 1`, positive strikes, non-negative bids, positive asks, no crossed book, days in `[7, 120]`, finite normalized prices. `QuoteFilter.permissive()` only sets `min_volume` to 0. On the 14-day expiry of 2017-07-05 the default keeps 86 rows; relaxing volume keeps 232. Changing the filter changes the sample and belongs in the specification.

### `from_quotes` and `from_frames`

For one day and one expiry, use `CrossSection.from_quotes`. Required: `date`, `expiry`, `forward`. Defaults are `rate=0.0`, `strike_scale=1.0` (**no** automatic guess of 1000), `am_settlement=0`; the filter still applies a volume floor of 1. `column_map` must be a `ColumnMap` object, not a raw dictionary.

For a multi-day panel use `Dataset.from_frames(options, forwards=..., rates=...)`. In-memory tables cannot `spec()`, so `frontier_series(..., workers>1)` falls back to a single process.

The script writes no files. Sample output:

- `nearest_date("2017-07-04")` → `2017-07-03`
- All three 14-day selectors agree: `expiry=2017-07-19`, `am_settlement=0`, `F=2431.978815`
- PM 16-day: `root=SPXW`, 95 quotes; AM 16-day: `root=SPX`, 150 quotes
- Unspecified settlement selects AM, `ambiguous_settlement=True`
- `from_quotes` has `forward_source` equal to `user`

## Options

| Task | Call |
|---|---|
| Afternoon session only | `am_settlement=0` |
| A specific root | `root="SPXW"` |
| Target 30 days | `target_days=30, window=(20, 45)` |
| Override forward / rate | `cross_section(..., forward=2432.0, rate=0.01)` |
| Relax volume | `Dataset.from_directory(..., quote_filter=QuoteFilter.permissive())` |
| One quote per strike | `QuoteFilter(one_quote_per_strike=True)` |
| Your own column names | `ColumnMap.detect(df.columns, overrides={"strike": "k", ...})` |

## Common errors

- **`Dataset(root)` instead of `from_directory`.** The constructor looks for option tables in that folder only: it does not enter `RAW_DATA/` and does not attach forwards, rates, or spot.
- **Weekend observation date.** `NoSuchDateError` reports the nearest trading days on either side. Use `nearest_date` and record the change; do not silently rewrite the date.
- **Every strike looks 1000× too large.** `describe()["strike_scale"]` should be 1000. `from_quotes` defaults to `strike_scale=1.0`; OptionMetrics raw tables must pass 1000, or be divided first.
- **Date `07/05/2017`.** Default is month/day/year; day-first requires `dayfirst=True`.
- **Applying a `--window` habit to `days=`.** In Python, `window` affects only `target_days`.
- **The rate series has no observation on or before the quote date.** Raises `ValueError` and mentions `rates=0.0`. `rates=None` is an explicit \(r=0\), not an automatic repair for missing data.
