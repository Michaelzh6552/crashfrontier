# Dates, `dayfirst`, strike scaling, AM/PM

Parsers: `parse_dates`, `detect_date_format`, `detect_strike_scale`,
`parse_settlement` (`crashfrontier/data/parse.py`).
`Dataset.describe()["date_format"]` and `["strike_scale"]` are the first
place to check a bad read.

## Dates become midnight that calendar day

Timezones are stripped. `2021-07-21 15:30:00+00:00` → `2021-07-21`
(`test_parse_dates_datetime64_normalizes_and_drops_tz`).

Order:

1. Already datetime: normalize only.
2. Numeric, or **all** values are integer-valued numeric strings (digits,
   optionally followed by `.0`, `.00`, and so on):
   - every value ∈ `[19000101, 21001231]`: `%Y%m%d` (warning: `interpreting numeric dates as YYYYMMDD`).
   - every value ∈ `[5000, 30000]`: SAS days, origin `1960-01-01`.
   - every value ∈ `[30000, 80000]`: Excel serial, origin `1899-12-30`.
   - else: `SchemaError: unrecognized numeric date values`.
3. Strings: up to 200 nonempty unique samples; the **first** format in the tuple that parses every sample wins. `%b` is applied after `.str.title()` (`21JUL2021` / `21jul2021` both work).
4. None match: `pd.to_datetime(..., errors="raise", dayfirst=bool(dayfirst))`, plus a warning.

`DATE_FORMATS` (`test_date_formats_tuple_matches_spec`):

```text
%Y-%m-%d
%Y%m%d
%m/%d/%Y
%d/%m/%Y
%Y/%m/%d
%d%b%Y
%d-%b-%Y
%d-%b-%y
%b %d, %Y
%Y-%m-%d %H:%M:%S
%Y-%m-%dT%H:%M:%S
%m/%d/%y
%d.%m.%Y
%Y.%m.%d
%m-%d-%Y
%d-%m-%Y
```

`detect_date_format` returns the token that will be used, or `None` for
native datetime. The three numeric cases are `"%Y%m%d"`, `"sas_days"`,
`"excel_serial"`.

## `dayfirst`: ambiguous dates default to month-day-year

When both `%m/%d/%Y` and `%d/%m/%Y` accept the samples (and the `-` pair):

| `dayfirst` | Behavior |
|---|---|
| `True` | day/month/year |
| `False` | month/day/year |
| `None` | any sample with first field `> 12` → day first; any sample with second field `> 12` → month first; all fields ≤12 → **month first**, warning `assuming month-first` |

`05/06/2021` defaults to 2021-05-06, not 5 June
(`test_parse_dates_ambiguous_defaults_to_month_first`). A first field above
12 cannot be a month, so `%m/%d/%Y` fails and the string is read as
day/month/year. `13/05/2021` and `17/01/2024` both parse that way with
`dayfirst` left at `None`.

`Dataset` and `from_quotes` apply the same `dayfirst` to the `date`/`expiry`
**arguments** and to date columns in the table.

## Strike scaling: `strike_scale`

`canonicalize` does `strike = strike / strike_scale`. Candidates are only
`(1, 10, 100, 1000)`.

`detect_strike_scale(strikes, reference_price)`:

- Finite positive reference: choose `s` that minimizes `|log((median(strike)/s) / F_ref)|`.
- Non-finite reference: median strike `> 100000` → `1000`, else `1`, plus a warning that you should set the scale explicitly.

`Dataset(strike_scale="auto")` (the default) computes once on the **first**
metadata need and caches it (`_ensure_meta`, `test_strike_scale_detected_once`).
Reference-price chain (`_strike_reference`, using `available_dates()[0]`):

1. median of all forwards that day in the forward table (across expiries and settlements);
2. else nonempty median of the options `forward_price` column;
3. else spot that day;
4. else NaN → the “no reference” branch.

OptionMetrics `strike_price` is typically index×1000. With a forward or
spot in the usual index-point range, detection is `1000.0`.

`CrossSection.from_quotes` **does not** use this chain. Default is `1.0`.

Any other string: `SchemaError: strike_scale must be 'auto' or a number`.

Probe for a wrong scale: `quotes["x"].median()` on a cross-section should be
near 1. Near 1000 or 0.001 means you divided too little or too much. **Do
not “fix” it by editing `K`.**

## AM/PM settlement {#am-pm-settlement}

`am_settlement=1` is AM, `0` is PM. The two settlements on the same calendar
expiry are two cross-sections: separate `expiries()` rows, separate
`cross_sections` objects, **never merged**
(`test_expiries_keeps_am_pm_separate`,
`test_cross_sections_does_not_mix_settlement`).

`parse_settlement(values, root)`:

- No settlement column: `root == "SPX"` → 1; everything else (including `SPXW`, `""`) → 0.
- Column present: nonzero numeric → 1, 0 → 0 (`2` counts as 1); text `AM/A/1/TRUE/Y` → 1, `PM/P/0/FALSE/N` → 0 (case-insensitive). A missing cell falls back to the root rule.
- Unrecognized: `SchemaError: unrecognized settlement values`.

`am_settlement` omitted while both AM and PM hit: warning
`ambiguous AM/PM settlement`; pick the larger `n_put_strikes`; tie sorts
`am_settlement` ascending, **PM (0) first**.
`summary()["ambiguous_settlement"] is True`. Record which side was chosen.
For a reproducible pick, pass `am_settlement=0` or `1`.

On a typical SPX IvyDB extract, `SPX` monthly expiries are AM and `SPXW`
weeklies are PM. The same Friday can appear twice; the rows stay separate.

## Parse checks

```python
import logging
import pandas as pd
from crashfrontier.data.parse import (
    DATE_FORMATS,
    detect_date_format,
    detect_strike_scale,
    parse_dates,
    parse_settlement,
)

assert "%d%b%Y" in DATE_FORMATS

# 1) unambiguous formats
assert parse_dates(pd.Series(["21JUL2021"])).iloc[0] == pd.Timestamp("2021-07-21")
assert detect_date_format(pd.Series(["21JUL2021"])) == "%d%b%Y"
assert detect_date_format(pd.Series([20210721])) == "%Y%m%d"

# 2) dayfirst
assert parse_dates(pd.Series(["05/06/2021"]), dayfirst=True).iloc[0] == pd.Timestamp("2021-06-05")
assert parse_dates(pd.Series(["05/06/2021"]), dayfirst=False).iloc[0] == pd.Timestamp("2021-05-06")
logging.getLogger("crashfrontier").setLevel(logging.WARNING)
ambiguous = parse_dates(pd.Series(["05/06/2021"]))
assert ambiguous.iloc[0] == pd.Timestamp("2021-05-06")

# 3) strike_scale
assert detect_strike_scale(pd.Series([3_900_000.0, 4_000_000.0, 4_100_000.0]), 4000.0) == 1000.0
assert detect_strike_scale(pd.Series([3900.0, 4000.0, 4100.0]), 4000.0) == 1.0
assert detect_strike_scale(pd.Series([4_000_000.0]), float("nan")) == 1000.0
assert detect_strike_scale(pd.Series([4000.0]), float("nan")) == 1.0

# 4) settlement
root = pd.Series(["SPX", "SPXW", ""])
assert parse_settlement(None, root).tolist() == [1, 0, 0]
assert parse_settlement(pd.Series(["AM", "PM", "N"]), root).tolist() == [1, 0, 0]
print("parse checks OK")
```

All asserts pass. `05/06/2021` without `dayfirst` is 6 May.

## Check on a small `Dataset`

```python
from pathlib import Path
import tempfile
import pandas as pd
from crashfrontier import Dataset

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    opt = tmp / "opt.csv"
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
        }
    ).to_csv(opt, index=False)
    forwards = pd.DataFrame(
        {
            "date": ["2021-07-21"],
            "expiration": ["2021-08-04"],
            "AMSettlement": [0],
            "ForwardPrice": [4000.0],
        }
    )
    ds = Dataset(
        opt,
        forwards=forwards,
        rates=0.05,
        cache_dir=tmp / "CACHE",
    )
    info = ds.describe()
    assert info["date_format"] == "%Y-%m-%d"
    assert info["strike_scale"] == 1000.0
    rows = ds.option_rows("2021-07-21")
    assert rows["date"].dt.normalize().eq(pd.Timestamp("2021-07-21")).all()
    assert rows["strike"].tolist() == [3900.0, 4100.0]
    cs = ds.cross_section("2021-07-21", days=14)
    assert 0.8 < float(cs.quotes["x"].median()) < 1.2
    print("date/strike OK", info["date_format"], info["strike_scale"], float(cs.quotes["x"].median()))
```

On a real extract, print `ds.describe()["date_format"]` and
`["strike_scale"]` first (`CRASHFRONTIER_DATA`). For a column like
`07/05/2017`: if you want 5 July and the format is `%m/%d/%Y`, rebuild
`Dataset` with `dayfirst=True`. Do not later edit `K` or the strike column.
