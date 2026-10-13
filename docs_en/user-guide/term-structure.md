# Term structure: one day, several expiries

`term_structure` computes several expiries on one observation date and returns a summary table plus a collection of `Frontier` objects. The script below uses observation date **2017-07-05** and the risk-neutral put wing at target 7, 14, and 30 days.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `term_structure_run.py` and run `python term_structure_run.py`.

```python
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import Dataset, plot_term_structure, save_figure, term_structure


def main() -> None:
    output = Path("outputs/user-guide/term-structure")
    ds = Dataset.from_directory(Path(os.environ["CRASHFRONTIER_DATA"]))
    table, fronts = term_structure(
        ds,
        "2017-07-05",
        tenors=(7, 14, 30),
        K=0.90,
        gamma=0,
    )
    print(table[["target_days", "date", "expiry", "days", "am_settlement",
                 "quote_set", "prob_lo", "prob_hi", "depth_lo", "depth_hi", "error"]].to_string(index=False))
    print("successful tenors:", sorted(fronts))

    csv_path = Path("outputs/user-guide/term-structure.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    save_figure(plot_term_structure(table), output, formats=("png", "pdf"))
    print("wrote", output.resolve(), "and", csv_path.resolve())


if __name__ == "__main__":
    main()
```

## How to read the result

```text
term_structure(ds, date, *, tenors=(7, 14, 30, 60, 90), window_fraction=0.35,
               K=0.9, gamma=0, am_settlement=None, **frontier_kwargs)
```

For each target day count \(T\), the window is

\[
\bigl[\max(d_{\min},\mathrm{round}(T(1-f))),\ \min(d_{\max},\mathrm{round}(T(1+f)))\bigr]
\]

where \(f=\) `window_fraction` (default 0.35) and \(d_{\min},d_{\max}\) come from the dataset's `QuoteFilter`. Then `ds.cross_section(..., target_days=T, window=(low, high))` is called.

When two targets select the same expiry, only the target closer to that expiry's actual `days` is kept; the other row gets `error="expiry already used"`. Failed rows stay in the table, do not enter the `fronts` dictionary, and are dropped from the figure.

The table has two extra columns beyond a single `summary()`: `target_days` and `error`. The default quote set at \(\gamma=0\) is still `put_wing`. For a physical version pass `quote_set="put_wing"` and `support_cap=5.0` explicitly.

`plot_term_structure(table)` draws two panels: the crash-probability band (%) and the conditional-depth band (%). A wider band means looser constraints at that tenor. The horizontal axis is `days`.

The 7 / 14 / 30 tenors all succeed:

| target_days | expiry | days | prob_lo | prob_hi |
|---:|---|---:|---:|---:|
| 7 | 2017-07-12 | 7 | 0.000024 | 0.002405 |
| 14 | 2017-07-19 | 14 | 0.000657 | 0.006926 |
| 30 | 2017-08-04 | 30 | 0.008757 | 0.015066 |

The script writes `outputs/user-guide/term-structure.png` / `.pdf` and `outputs/user-guide/term-structure.csv`.

<figure class="cf-example-figure">
  <img src="../assets/images/examples/07_term_structure.png" alt="Crash probability and conditional-depth term structure" loading="lazy">
  <figcaption>
    2017-07-05, γ=0, K=0.90, target tenors 7, 14, and 30 days. The package does not include these quotes.
  </figcaption>
</figure>

With the default `tenors=(7, 14, 30, 60, 90)`, whether 60 and 90 find a match depends on the expiries in your extract. If two tenors land on the same expiry you will see `expiry already used`.

## Options

| Task | Call |
|---|---|
| Full default tenors | Drop `tenors=...` |
| Narrower window | `window_fraction=0.20` |
| Afternoon session only | `am_settlement=0` |
| Physical | `gamma=2, quote_set="put_wing", support_cap=5.0` |
| Other observation date | Start from `ds.expiries("date")` |

## Common errors

- **Reading the horizontal axis as target_days.** The figure uses actual `days`. Keep both columns when reading the table.
- **Expecting failed rows on the figure.** They are dropped. Read the `error` column first.
- **Placing this figure next to a 2020-03-25 physical figure as “the same term structure.”** The sample here is 2017-07-05, \(\gamma=0\).
- **Passing `days=30` as a term structure.** `term_structure` has no such argument; it selects cross-sections from `tenors`.
