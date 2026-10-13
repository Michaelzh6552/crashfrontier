# `QuoteFilter` and the filter report

The quote filter is a `frozen` dataclass. The default is `QuoteFilter()`,
which equals `QuoteFilter.default()`. `QuoteFilter.permissive()` changes
**only** `min_volume` to `0.0`; everything else is unchanged
(`test_default_and_permissive_filters`).

```python
QuoteFilter(
    min_volume=1.0,
    min_open_interest=0.0,
    positive_strike=True,
    nonnegative_bid=True,
    positive_ask=True,
    uncrossed=True,          # ask >= bid
    min_days=7,
    max_days=120,
    min_width=0.0,
    require_finite=True,
    cp_flags=("C", "P"),
    one_quote_per_strike=False,
)
```

`Dataset` and `CrossSection.from_quotes` both attach this object by default.
To include zero volume: `QuoteFilter.permissive()` or
`QuoteFilter(min_volume=0)`, and record the rule. Zero-volume rows change
the constraint set.

## Two stages, rows dropped in order

`build_cross_section`: `apply_rows` first (canonical rows), then normalize,
then `apply_normalized`. The `filter_report` merges both stages. Rule names
are `ROW_RULES` + `NORMALIZED_RULES`, plus `volume_unavailable` and `kept`.

`kept` is the row count **after the normalized stage**, equal to
`len(cs.quotes)`
(`test_synthetic_optionmetrics_from_quotes_matches_build`).
`cs.summary()` **does not** include filter counts.

### `apply_rows` (canonical rows)

Order is fixed. Counts are **rows newly dropped at that step**
(`test_apply_rows_counts_and_does_not_mutate`; the input is not mutated):

1. `cp_flag`: `cp_flag` must be in `cp_flags` (default C/P).
2. `positive_strike`: `strike > 0`.
3. `nonnegative_bid`: `bid >= 0`.
4. `positive_ask`: `ask > 0`.
5. `uncrossed`: `ask >= bid`.
6. `min_volume`: only if `min_volume > 0` **and** the volume column is not entirely missing. Condition `volume >= min_volume`. Volume-0 rows are dropped under the default filter.
7. `min_open_interest`: only if the threshold is `> 0`.
8. `days_window`: `min_days ≤ days ≤ max_days`.

`volume_unavailable`: `1` when the volume column is absent or all-NaN; then
`min_volume` is **skipped** (its count stays 0) and the other rules still
run (`test_volume_all_missing_skips_min_volume`). Some NaN and some
numbers: not unavailable; NaN rows fail `volume >= 1`.

`apply_rows`’s own `kept` is the count after the row filter and before
normalization; when merged into `CrossSection.filter_report` it is
overwritten by the normalized-stage `kept`.

### `apply_normalized`

1. `finite_normalized` (`require_finite`): `x`, `norm_l`, `norm_u` finite and `norm_l <= norm_u`.
2. `min_width`: only if `min_width > 0`, `norm_u - norm_l >= min_width`.
3. `one_quote_per_strike`: for each `(cp_flag, strike)`, keep the narrowest interval; ties take the first `quote_id` in stable sort. Call and put do not affect each other.

## Relation to expiry selection

`Dataset._candidates` **also** filters the `expiries()` table with
`quote_filter.min_days` / `max_days`. Under the defaults, a `days=3` expiry
is neither a candidate nor kept by `apply_rows` (`days_window`).

`expiries()` `n_put_strikes` counts unique put strikes **after
`apply_rows`** with \(\log(\kappa/F)\in[-0.35, 0.05]\). When `days` is
outside the window, `n_rows>0` but `n_put_strikes` is often 0. A 0-, 2-,
or 5-day expiry under the default window is in that state.

Selecting ultra-short expiries means setting `min_days` to 0 — a change of
rule, to be written into the method note, not “fixing the data”.

## Report keys

```text
cp_flag
positive_strike
nonnegative_bid
positive_ask
uncrossed
min_volume
min_open_interest
days_window
finite_normalized
min_width
one_quote_per_strike
volume_unavailable
kept
```

## Drop-count example

```python
import pandas as pd
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.cross_section import build_cross_section

DATE = pd.Timestamp("2021-07-21")
EXPIRY = pd.Timestamp("2021-08-04")

def rows(records):
    base = dict(
        date=DATE,
        expiry=EXPIRY,
        cp_flag="P",
        strike=90.0,
        bid=5.0,
        ask=5.2,
        volume=10.0,
        open_interest=10.0,
        impl_volatility=0.18,
        quote_id="q",
        am_settlement=0,
        symbol="SPXW",
        root="SPXW",
        forward_price=float("nan"),
        expiry_indicator="w",
        days=14,
    )
    out = []
    for i, rec in enumerate(records):
        item = dict(base)
        item.update(rec)
        item["quote_id"] = rec.get("quote_id", f"q{i}")
        out.append(item)
    return pd.DataFrame(out)

frame = rows(
    [
        {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
        {"cp_flag": "P", "strike": 90.0, "bid": 4.0, "ask": 4.2, "volume": 0.0},
        {"cp_flag": "C", "strike": -1.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
        {"cp_flag": "C", "strike": 100.0, "bid": 2.0, "ask": 1.0, "volume": 10.0},
        {"cp_flag": "X", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
        {"cp_flag": "C", "strike": 100.0, "bid": -0.5, "ask": 1.2, "volume": 10.0},
        {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 0.0, "volume": 10.0},
        {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0, "days": 3},
    ]
)
kept, report = QuoteFilter.default().apply_rows(frame)
assert report["cp_flag"] == 1
assert report["positive_strike"] == 1
assert report["nonnegative_bid"] == 1
assert report["positive_ask"] == 1
assert report["uncrossed"] == 1
assert report["min_volume"] == 1
assert report["days_window"] == 1
assert report["volume_unavailable"] == 0
assert report["kept"] == 1
assert kept["quote_id"].tolist() == ["q0"]

zero = rows([{"volume": 0.0}, {"volume": 10.0}])
_, dft = QuoteFilter.default().apply_rows(zero)
_, perm = QuoteFilter.permissive().apply_rows(zero)
assert dft["kept"] == 1 and perm["kept"] == 2

cs = build_cross_section(
    kept,
    date=DATE,
    expiry=EXPIRY,
    am_settlement=0,
    forward=100.0,
    forward_source="user",
    rate=0.0,
    rate_source="user",
    spot=float("nan"),
    quote_filter=QuoteFilter.default(),
)
assert cs.filter_report["kept"] == 1
assert cs.filter_report["kept"] == len(cs.quotes)
print("filter report", cs.filter_report)
```

Of 8 rows, only `q0` remains; permissive keeps zero volume;
`cs.filter_report["kept"]==1`.

## On a real cross-section

Requires a folder you already hold (`CRASHFRONTIER_DATA`).

```python
import os
from pathlib import Path
from crashfrontier import Dataset

root = Path(os.environ["CRASHFRONTIER_DATA"])
ds = Dataset.from_directory(root)
cs = ds.cross_section(ds.available_dates()[-1], target_days=30)
print(cs.filter_report)
assert cs.filter_report["kept"] == cs.summary()["n_quotes"]
assert cs.filter_report["kept"] > 0
```

`min_volume` and `kept` follow the extract and the quote filter. Do not
edit volume in the CSV to hit a previous kept count.

## Infeasible solves

`InfeasibleCrossSectionError` means: under the current \(\gamma\), quote
set, and support, the **existing bid/ask intervals** have no compatible
measure. It is not a hint that “two rows are missing”.

Editing quotes to force feasibility — setting `ask` to `max(ask, bid)`,
writing fake `volume` onto dropped rows, inserting Black-76 synthetic rows to
“complete” the cross-section, or tightening `norm_u` — invalidates the result.

Read the `filter_report` and check that `kept==0` is not a column mis-map;
check `strike_scale`, the forward, and whether AM/PM were mixed into one
table; pass an explicit `QuoteFilter` (and record it); change the
`quote_set` or the day; or mark the day as failed and continue the time
series (`on_error`).

The quote filter exists to drop **obviously invalid** rows (negative
strike, crossed book, zero volume if you chose to exclude it), not to edit
the market into a solvable one.
