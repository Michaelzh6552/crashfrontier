# `Dataset.from_frames`

Use this for multi-day pandas tables already in memory. The CLI cannot read
a DataFrame. The implementation is the constructor:

```python
@classmethod
def from_frames(cls, options: pd.DataFrame, *, forwards=None, rates=None, spot=None, **kwargs):
    return cls(options, forwards=forwards, rates=rates, spot=spot, **kwargs)
```

Every keyword on the
[constructor](from-directory.md#constructor-dataset-is-not-from_directory)
therefore applies: `column_map`, `strike_scale`, `dayfirst`, `cp_values`,
`rate_unit`, `forward_sources`, `quote_filter`, `cache_dir`,
`memory_limit`.

`from_frames` accepts in-memory DataFrames. File paths are read by
`Dataset(...)` or `Dataset.from_directory(...)`.

## What each argument is

| Argument | Accepts | Notes |
|---|---|---|
| `options` | **must** be a `DataFrame` | columns go through aliases. Multiple days and expiries are fine; slice later with `cross_section` |
| `forwards` | `None` / path / `DataFrame` / existing `ForwardTable` | `None` drops `table` from the forward chain |
| `rates` | `None` / path / `DataFrame` / **scalar** | see below |
| `spot` | `None` / path / `DataFrame` / existing `SpotTable` | exact-date lookup; missing days are NaN; no forward fill |

`rates` dispatch (`open_rate_source`, `test_open_rate_source_dispatch`):

| Passed in | Resulting `name` | Unit |
|---|---|---|
| `None` | `"zero"` (`r=0`, warning on first use) | — |
| int/float such as `0.053` | `"constant"`, used as a decimal | **`rate_unit` is ignored** |
| table with a `days` column | `"zero_curve"` | **always percent /100**; `rate_unit` is ignored |
| two columns, date + value (e.g. FRED `DGS1MO`) | `"series"` | `rate_unit="percent"` → /100; `"decimal"` → no divide |
| `True`/`False` | `SchemaError: boolean is not a rate source` | — |

Option columns still go through `ColumnMap.detect`. OptionMetrics-style
`exdate` / `strike_price` / `best_bid` / `best_offer` work as-is.

`strike_scale` still defaults to `"auto"`. If an in-memory options table is
paired with a forward table, the reference price is the median forward on
the **first observation date** (same as the file path). If your own strikes
are already in index points, pass `strike_scale=1.0`.

## `spec()` is unavailable; extra workers fall back to one process

`Dataset.spec()` raises
`ValueError: in-memory option source cannot be serialized` when the option
source is an in-memory table (`test_spec_rejects_frames`).

`frontier_series(..., workers>1)` calls `ds.spec()` first; on this
`ValueError` it logs a warning (message contains `in-memory` and
`workers=1`) and runs in one process
(`test_memory_source_falls_back_to_one_worker`). A DataFrame cannot be
pickled to a child process.

## In-memory panel

```python
import math
import numpy as np
import pandas as pd
from crashfrontier import Dataset, QuoteFilter

DATE = "2024-01-17"
EXPIRY = "2024-02-16"
FORWARD = 4790.5
RATE = 0.053  # already a decimal
SPOT = 4785.0

date = pd.Timestamp(DATE)
expiry = pd.Timestamp(EXPIRY)
days = int((expiry - date).days)
assert days == 30

strikes = np.round(np.linspace(0.85 * FORWARD, 1.10 * FORWARD, 8) / 5.0) * 5.0
rows = []
for k in strikes:
    for cp, bid, ask in (("P", 20.0, 21.0), ("C", 15.0, 16.0)):
        rows.append(
            {
                "date": DATE,
                "exdate": EXPIRY,
                "cp_flag": cp,
                "strike_price": float(k),
                "best_bid": bid,
                "best_offer": ask,
                "volume": 5,
                "am_settlement": 0,
                "symbol": "SYN",
            }
        )
options = pd.DataFrame(rows)
forwards = pd.DataFrame(
    {
        "date": [DATE],
        "expiration": [EXPIRY],
        "AMSettlement": [0],
        "ForwardPrice": [FORWARD],
    }
)
# FRED style: 5.3 in the table means 5.3%
rates = pd.DataFrame({"observation_date": [DATE], "DGS1MO": [RATE * 100.0]})
spot = pd.DataFrame({"Date": [DATE], "Price": [SPOT]})

ds = Dataset.from_frames(
    options,
    forwards=forwards,
    rates=rates,
    spot=spot,
    strike_scale=1.0,
    rate_unit="percent",
)
info = ds.describe()
assert info["options"]["kind"] == "frame"
assert info["options"]["files"] == []
assert info["options"]["n_dates"] == 1
assert info["date_format"] == "%Y-%m-%d"
assert info["strike_scale"] == 1.0
assert info["forwards"] == "table"
assert info["rates"] == "series"
assert info["spot"] is True

cs = ds.cross_section(DATE, days=30, am_settlement=0)
assert cs.forward == FORWARD
assert cs.forward_source == "table"
assert math.isclose(cs.rate, RATE)
assert cs.rate_source == "series"
assert cs.spot == SPOT
assert cs.days == 30
assert set(cs.quotes["cp_flag"]) <= {"C", "P"}
print(ds.describe())
print(cs.summary())
print(cs.filter_report)

try:
    ds.spec()
    raise SystemExit("from_frames spec() should fail")
except ValueError as exc:
    assert "in-memory" in str(exc)
    print("spec() correctly rejected:", exc)
```

`kind=="frame"`, `files==[]`, forward/rate/spot sources as above;
`spec()` raises `in-memory`. Bid/ask in the script are placeholder constants
for the read path, **not** a publishable market. Replace them with real
bid/ask in your own work; do not edit those two columns to make the LP
feasible.

## Four rate sources, side by side

```python
import pandas as pd
from crashfrontier import Dataset

options = pd.DataFrame(
    {
        "date": ["2021-07-21"],
        "expiry": ["2021-08-04"],
        "cp_flag": ["P"],
        "strike": [3900.0],
        "bid": [10.0],
        "ask": [10.5],
        "volume": [3],
        "forward_price": [4000.0],
    }
)

def section(rates, **kwargs):
    ds = Dataset.from_frames(
        options, rates=rates, strike_scale=1.0, quote_filter=__import__(
            "crashfrontier"
        ).QuoteFilter.permissive(),
        **kwargs,
    )
    cs = ds.cross_section("2021-07-21", days=14)
    return cs.rate, cs.rate_source

assert section(None) == (0.0, "zero")
assert section(0.053) == (0.053, "constant")
assert section(pd.DataFrame({"date": ["2021-07-21"], "rf": [5.3]}), rate_unit="percent") == (
    0.053,
    "series",
)
assert section(pd.DataFrame({"date": ["2021-07-21"], "rf": [0.053]}), rate_unit="decimal") == (
    0.053,
    "series",
)
curve = pd.DataFrame(
    {"date": ["2021-07-21", "2021-07-21"], "days": [7, 365], "rate": [5.0, 7.0]}
)
r_curve, name = section(curve, rate_unit="decimal")  # unit is ignored
assert name == "zero_curve"
assert abs(r_curve - (0.05 + (0.07 - 0.05) * (14 - 7) / (365 - 7))) < 1e-12
print("rate dispatch OK", r_curve)
```

The four `rate_source` names match the table above. A zero curve is still
read as percent when `rate_unit="decimal"` (5.0 → 0.05). If your zero-curve
nodes are already decimals, do not use this table: pass `rates=0.053` as a
constant, or multiply the column by 100 first. Details:
[forwards-rates.md](forwards-rates.md).

## Choosing among `from_quotes` and the file entry points

- One day, one expiry, and you already have the forward: [`CrossSection.from_quotes`](from-quotes.md). No `Dataset` wrapper.
- Several days, and you need `expiries` / `nearest_date` / `frontier_series`: `from_frames`.
- Tables on disk that may exceed 512 MiB: do not `pd.read_csv` then `from_frames`. Use [`from_directory`](from-directory.md) or constructor paths so the [byte-offset date index](large-files.md) can apply.

A runnable synthetic-market example is `examples/custom_data.py` in the
repository (`from_frames` vs `from_quotes`). Bid/ask there come from
Black-76; they are teaching synthetics, not real OptionMetrics rows.
