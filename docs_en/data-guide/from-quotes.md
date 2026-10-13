# `CrossSection.from_quotes`

Use this when you already have **one observation date, one expiry** of option
quotes and you know the forward (rate and spot are optional). It does not
scan a directory and does not detect `strike_scale`.

On the same canonical rows, `from_quotes` and `Dataset.cross_section` should
produce the same `quotes` (`test_from_quotes_matches_build_cross_section`,
`test_dataset_cross_section_matches_from_quotes`). Metadata differs:
`forward_source` / `rate_source` are `"user"` here.

## Signature and defaults

```python
CrossSection.from_quotes(
    quotes: pd.DataFrame,
    *,
    date,                      # required
    expiry,                    # required
    forward: float,            # required
    rate: float = 0.0,
    spot: float = float("nan"),
    am_settlement: int = 0,
    column_map: ColumnMap | None = None,  # ColumnMap or None only
    strike_scale: float = 1.0,            # not "auto"
    dayfirst: bool | None = None,
    cp_values: dict[str, str] | None = None,
    quote_filter: QuoteFilter = QuoteFilter(),
) -> CrossSection
```

Three differences from `Dataset`:

1. There is **no** `strike_scale="auto"`. Default is `1.0`. If OptionMetrics `strike_price` is index×1000, pass `strike_scale=1000`.
2. `column_map` **cannot** be a dict. Passing `{"date": "trade_date"}` becomes `AttributeError` when `.date` is read. Call `ColumnMap.detect(columns, overrides=...)` first.
3. The quote filter is still the default `QuoteFilter()`: `volume >= 1`, calendar days `[7, 120]`. If the table has no volume column, the volume rule is skipped (`volume_unavailable=1`) and rows are not all dropped.

The `date` / `expiry` **arguments** also go through
`parse_dates(..., dayfirst=)`
(`test_from_quotes_applies_dayfirst_to_date_arguments`).
`05/07/2021` + `dayfirst=True` is 5 July, not 7 May.

## When the table has no date/expiry columns

If `column_map is None` and the header (after normalization) intersects
neither the `date` nor the `expiry` alias sets, the arguments are written
into `date` and `expiry` columns, then `ColumnMap.detect` runs
(`_fill_missing_date_expiry`).

If you pass a `ColumnMap` and the mapped `date` / `expiry` source names are
absent, those columns are filled under the **source names**. Do not construct
a `ColumnMap(...)` that is missing required fields.

After canonicalize, only rows whose `date` and `expiry` match the arguments
are kept. If the table has a settlement column, it is filtered by the
`am_settlement` argument; if not, the whole column is set to that argument
(default 0).

Discount: `discount = exp(-rate * days / 365)`,
`days = (expiry - date).days`.

## Single-day table

```python
import math
import pandas as pd
from crashfrontier import ColumnMap, CrossSection, QuoteFilter

raw = pd.DataFrame(
    {
        "cp_flag": ["P", "C"],
        "strike": [4300.0, 4800.0],
        "bid": [10.0, 11.0],
        "ask": [11.0, 12.0],
        "volume": [5, 8],
    }
)

cs = CrossSection.from_quotes(
    raw,
    date="2024-01-17",
    expiry="2024-02-16",
    forward=4790.5,
    rate=0.053,
    spot=4785.0,
    am_settlement=0,
    strike_scale=1.0,
)
assert cs.date == pd.Timestamp("2024-01-17")
assert cs.expiry == pd.Timestamp("2024-02-16")
assert cs.days == 30
assert cs.am_settlement == 0
assert cs.forward_source == "user"
assert cs.rate_source == "user"
assert cs.forward == 4790.5
assert math.isclose(cs.discount, math.exp(-0.053 * 30 / 365))
assert cs.quotes["strike"].tolist() == [4300.0, 4800.0]
assert list(cs.quotes.columns)[:5] == ["quote_id", "symbol", "root", "cp_flag", "strike"]
assert "norm_l" in cs.quotes.columns
print(cs)
print(cs.summary())
print(cs.filter_report)
```

`days==30`, `forward_source=="user"`, strikes stay 4300/4800
(`strike_scale=1`). `filter_report["kept"]==2`.

## OptionMetrics slice + `strike_scale=1000`

```python
import pandas as pd
from crashfrontier import ColumnMap, CrossSection, QuoteFilter

om = pd.DataFrame(
    {
        "date": ["2021-07-21", "2021-07-21"],
        "exdate": ["2021-08-04", "2021-08-04"],
        "cp_flag": ["P", "C"],
        "strike_price": [3900000.0, 4100000.0],
        "best_bid": [12.0, 9.0],
        "best_offer": [12.4, 9.3],
        "volume": [10, 8],
        "optionid": [11, 12],
        "am_settlement": [0, 0],
        "symbol": ["SPXW 210804P3900000", "SPXW 210804C4100000"],
    }
)
mapping = ColumnMap.detect(om.columns)
assert mapping.strike == "strike_price"
assert mapping.ask == "best_offer"

cs = CrossSection.from_quotes(
    om,
    date="2021-07-21",
    expiry="2021-08-04",
    forward=4000.0,
    rate=0.06,
    strike_scale=1000.0,
    column_map=mapping,
    quote_filter=QuoteFilter.default(),
)
assert cs.quotes["strike"].tolist() == [3900.0, 4100.0]
assert cs.root == "SPXW"
assert cs.filter_report["min_volume"] == 0
print(cs.quotes[["cp_flag", "strike", "x", "norm_l", "norm_u"]])
```

Strikes become 3900/4100; `x` is near 0.975 and 1.025. If you omit
`strike_scale=1000`, `x` is near 1000 and the identification interval is
meaningless.

## `dayfirst` and `ColumnMap.detect`

```python
import pandas as pd
from crashfrontier import ColumnMap, CrossSection, QuoteFilter

raw = pd.DataFrame(
    {
        "trade_date": ["05/07/2021"],
        "maturity": ["19/07/2021"],
        "type": ["P"],
        "k": [90000.0],
        "bid_px": [5.0],
        "ask_px": [5.2],
        "volume": [10],
    }
)
mapping = ColumnMap.detect(
    raw.columns,
    overrides={
        "date": "trade_date",
        "expiry": "maturity",
        "cp_flag": "type",
        "strike": "k",
        "bid": "bid_px",
        "ask": "ask_px",
    },
)
cs = CrossSection.from_quotes(
    raw,
    date="05/07/2021",
    expiry="19/07/2021",
    forward=100.0,
    rate=0.053,
    column_map=mapping,
    strike_scale=1000.0,
    dayfirst=True,
    quote_filter=QuoteFilter.permissive(),
)
assert cs.date == pd.Timestamp("2021-07-05")
assert cs.expiry == pd.Timestamp("2021-07-19")
assert cs.quotes["strike"].tolist() == [90.0]
print(cs.date, cs.expiry, cs.days)
```

2021-07-05 → 2021-07-19, `days==14`. Without `dayfirst=True`,
`05/07/2021` is read as 7 May (month-day-year).

```python
# do not: pass only overrides
# ColumnMap(date="trade_date", strike="k")
# TypeError: missing expiry, cp_flag, bid, ask

# do not: pass a dict to from_quotes
# CrossSection.from_quotes(..., column_map={"date": "trade_date"})
# AttributeError: 'dict' object has no attribute 'date'
```

`Dataset(..., column_map={"date": "trade_date"})` is fine: the constructor
treats a dict as overrides. `from_quotes` has no such conversion.

`cp_values={"1": "C", "0": "P"}` is applied before letter codes. The CLI
has no switch for this.

## Normalized columns on `cs.quotes`

`build_cross_section` writes columns in a fixed order (`QUOTE_COLUMNS`):

```text
quote_id symbol root cp_flag strike bid ask volume open_interest impl_volatility
x log_moneyness norm_bid norm_ask norm_l norm_u role
```

- \(x=\kappa/F\); `log_moneyness = log(x)` when \(x>0\).
- `norm_bid = bid / (D F)`; `norm_ask` likewise.
- Call: `[norm_l, norm_u] = [norm_bid, norm_ask]`.
- Put: `[norm_l, norm_u] = [norm_bid+1-x, norm_ask+1-x]`.
- `role`: put and \(x\le 1\) → `otm_put`; call and \(x\ge 1\) → `otm_call`; otherwise `itm` (ATM counts as OTM).

When bid=ask at the same strike and parity holds, put and call share
`[norm_l, norm_u]` (`test_put_call_parity_matches_when_bid_equals_ask`).
That is a probe for a wrong forward/discount, not a reason to edit option
quotes so that parity “holds”.

`from_quotes` accepts any bid/ask. That does not mean you may uncross a book
by setting `ask = bid`, write fake `volume=1` on zero-volume rows (use
`QuoteFilter.permissive()` and record the rule), or invent wider/narrower
quotes after `InfeasibleCrossSectionError`. Infeasibility means **this real
book** has no compatible measure under the current model. Check
`filter_report`, `strike_scale`, the forward, and whether AM/PM sit in one
table; pick another day; or shrink the quote set.
