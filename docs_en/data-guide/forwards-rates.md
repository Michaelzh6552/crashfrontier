# Forward chain, rate units, spot

A cross-section needs a finite forward. Otherwise that `expiries()` row has
`forward_source="none"` and `cross_section` will not select it. The rate is
used for the discount \(D=\exp(-r\cdot\mathrm{days}/365)\) and for the
parity forward. Spot is optional: mainly the `strike_scale="auto"`
reference, a parity reference, and `cs.spot`.

Attach tables you already hold, or pass a constant rate / user forward.

## Default forward chain

`Dataset` defaults to `forward_sources=("table", "column", "parity")`.
Unknown name: `SchemaError`.

`Dataset._resolve_forward` **assembles** sources in that order, then
`resolve_forward` takes the first finite value and returns
`(value, source.name)`.

| Name | When it is in the chain | Implementation |
|---|---|---|
| `table` | a forward table was passed at construction | `ForwardTable`, key `(date, expiry, am_settlement)` |
| `column` | always attempted (needs a forward column on the option rows) | `ForwardColumn`: median `forward_price` by expiry and settlement |
| `parity` | a rate for that date and day count is available; a `ValueError` from the rate **skips** this link | `ParityForward` |

`cross_section(..., forward=F)` replaces the resolved forward on the selected
cross-section and sets `forward_source="user"`. Selection still starts from
`expiries()`, so this argument cannot make an expiry whose forward chain
failed eligible.

The whole chain fails: `MissingForwardError`, message contains
`tried [...]`. `expiries()` catches that and writes NaN / `"none"`; it does
not re-raise.

### Forward table

Columns: date (option `date` aliases), expiry (`expiration` / `exdate` /
`expiry` / `expiration_date` / `maturity`), forward (`ForwardPrice` /
`forward_price` / `forward` / `fwd` / `fwd_price` / `f`), optional
settlement (`AMSettlement` / `am_settlement` / `am_set_flag` /
`settlement`).

- Duplicate `(date, expiry, am_settlement)`: **keep last**.
- Lookup with settlement: exact match first; if that settlement is absent, fall back to rows for that `(date, expiry)`.
- Several rows and no settlement, or several matches: prefer `am_settlement==0` (PM), else the first row in the group (`_pm_or_first`).
- Missing key returns `None`, not an exception.

OptionMetrics `forward.csv` often has extra columns such as `secid`; they
are ignored. When that table classifies, `describe()["forwards"]=="table"`.

### `forward_price` column on the option file

`ForwardColumn` **ignores the observation date** and aggregates a median by
expiry (and settlement). If that column is all NaN for an expiry, that
expiry has no column forward. Test variant `yyyymmdd_gz_column_forward` has
no forward file, only the column; the chain lands on `"column"`.

### Parity forward

Needs at least **2** strikes that have both a call mid and a put mid.
Formula: \(F = \kappa + (C-P)/D\), \(D\) the discount. Take the median of
at most **9** strikes near the reference; the reference prefers a finite
spot, else the median of implied forwards. Crossed rows (`ask < bid`),
`ask<=0`, `bid<0` are excluded. Too few pairs → `None`.

Parity is the last link in the chain, not a reason to edit bid/ask to
“make” a forward. Prefer the forward table or the `forward_price` column
when they work.

## Rate units

`open_rate_source(rates, *, dayfirst=None, unit="percent")`.

| Source | `describe()["rates"]` | Unit |
|---|---|---|
| two columns, date + value | `"series"` | this source alone honors `rate_unit`: `percent` → /100, `decimal` → no divide. Sorted by date, `ffill`, same-day keep last. Takes the **last observation on or before** the observation date. Earlier → `ValueError: no rate observation on or before ...; pass rates=0.0 or a constant` |
| `days` column | `"zero_curve"` | **always** `rate/100`. A passed `rate_unit` is discarded. Linear interpolation in days; flat outside the endpoints. Curve date is likewise the latest on or before the observation date |
| scalar `0.053` | `"constant"` | already a decimal. `rate_unit` unused |
| `None` | `"zero"` | `r=0`; first call warns `no interest-rate data; using r = 0` |

FRED `DGS1MO` cell `0.97` means 0.97%. Default `rate_unit="percent"` →
`0.0097`.

If the series already stores `0.053` as a decimal, you must pass
`rate_unit="decimal"`, or it is divided by 100 again and becomes 5.3 bp.

If zero-curve nodes are already decimals: do not expect
`rate_unit="decimal"` to help (same `days/rate` table, both units, same
result). Convert to percent, or skip the zero curve and use a constant.

`Dataset.spec()` records `rate_unit`. `Dataset.from_spec` reuses that
value, and uses `"percent"` only when the key is absent.

## Spot

`SpotTable` looks up an exact date and returns NaN when that day is
missing. It does not forward-fill. A frame or path you pass yourself accepts
price aliases `close`, `price`, `adj_close`, `spindx`, `level`, `last`,
`value`, and `spot`.

`Dataset.from_directory` classifies by headers. Those price aliases do not
include the column name `spot`, so a file whose only price column is `Spot`
is not attached. See [formats-columns.md](formats-columns.md).

Vendor spot files are often `Date` + `Price` (sometimes with a BOM,
thousands separators, or `%m/%d/%Y`). A filename containing `vix` or `vx`
is never attached as spot, including when it is the only price-like file.

## Three links of the chain

```python
import pandas as pd
from crashfrontier import Dataset, QuoteFilter
from crashfrontier.data.forwards import ForwardTable, ForwardColumn, resolve_forward
from crashfrontier.errors import MissingForwardError

flt = QuoteFilter.permissive()

# --- table ---
table = ForwardTable.from_frame(
    pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "expiration": ["2021-08-20", "2021-08-20"],
            "AMSettlement": [1, 0],
            "ForwardPrice": [4021.0, 4019.0],
        }
    )
)
assert table.forward("2021-07-21", "2021-08-20", 1) == 4021.0
assert table.forward("2021-07-21", "2021-08-20", 0) == 4019.0
assert table.forward("2021-07-21", "2021-08-20", None) == 4019.0  # tie → PM
assert table.forward("2021-07-22", "2021-08-20", 0) is None

# --- column ---
col = ForwardColumn(
    pd.DataFrame(
        {
            "expiry": ["2021-08-04", "2021-08-04"],
            "am_settlement": [0, 0],
            "forward_price": [100.0, 102.0],
        }
    )
)
assert col.forward("2021-07-21", "2021-08-04", 0) == 101.0

# --- chain ---
value, name = resolve_forward([table, col], "2021-07-21", "2021-08-20", 1)
assert (value, name) == (4021.0, "table")
value, name = resolve_forward([table, col], "2021-07-21", "2021-08-04", 0)
assert (value, name) == (101.0, "column")
try:
    resolve_forward([table, col], "2021-07-21", "2021-10-01", 0)
    raise SystemExit("should miss")
except MissingForwardError as exc:
    assert "tried" in str(exc)

# --- unit on Dataset ---
options = pd.DataFrame(
    {
        "date": ["2021-07-21"],
        "exdate": ["2021-08-04"],
        "cp_flag": ["P"],
        "strike": [3900.0],
        "bid": [10.0],
        "ask": [10.5],
        "volume": [4],
        "forward_price": [4000.0],
        "am_settlement": [0],
        "symbol": ["SPXW"],
    }
)
ds = Dataset.from_frames(
    options,
    rates=pd.DataFrame({"observation_date": ["2021-07-21"], "DGS1MO": [6.0]}),
    strike_scale=1.0,
    rate_unit="percent",
    quote_filter=flt,
)
cs = ds.cross_section("2021-07-21", days=14)
assert cs.forward == 4000.0
assert cs.forward_source == "column"  # no forward table
assert cs.rate == 0.06
assert cs.rate_source == "series"
print("forward/rate OK", cs.forward_source, cs.rate, cs.discount)
```

Table lookup distinguishes AM/PM; unspecified settlement uses PM 4019; the
chain falls through to column when the table key is missing; without a
forward table, Dataset has `forward_source=="column"`; 6.0% → `rate==0.06`.

## Your extract (`CRASHFRONTIER_DATA`)

```python
import os
from pathlib import Path
from crashfrontier import Dataset

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
info = ds.describe()
print(info["forwards"], info["rates"], info["spot"])

cs = ds.cross_section(ds.available_dates()[-1], target_days=30)
print(cs.forward_source, cs.rate_source, cs.forward, cs.rate, cs.spot, cs.discount)
```

Levels follow your files and any FRED revision. If you attached a forward
table and a two-column rate series, the source names should be `table` and
`series`, not `zero` or `parity`.

If `forward_source` is unexpectedly `parity`: check whether the forward
file failed classification, whether several forward tables were turned off
with `forwards=None`, or whether that `(date, expiry, AM)` is absent from
the table. Do not edit option bid/ask to “help” parity.
