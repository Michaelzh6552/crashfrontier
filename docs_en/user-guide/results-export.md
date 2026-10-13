# Results, units, and export

`write_frontier` writes CSV/JSON; `save_figure` writes images. The script below exports the 2017 three-layer nest and converts raw decimals to percents and basis points (bp) with pandas. 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `results_export.py` and run `python results_export.py`.

```python
import os
from pathlib import Path

import pandas as pd

from crashfrontier import Dataset, binding_quotes, nested_frontiers, write_frontier
from crashfrontier.export import frontier_frames


def main() -> None:
    stem = Path("outputs/user-guide/results-export")
    ds = Dataset.from_directory(Path(os.environ["CRASHFRONTIER_DATA"]))
    cs = ds.cross_section("2017-07-05", target_days=14, window=(7, 21))
    fronts = nested_frontiers(
        cs,
        K=0.90,
        gamma=0,
        sets=("put_pair", "local_eight", "put_wing"),
    )
    certificate = binding_quotes(cs, 0.90, direction=(1.0, 0.0))
    paths = write_frontier(
        fronts,
        stem,
        formats=("csv", "json"),
        certificate=certificate,
    )
    print("wrote:")
    for path in paths:
        print(" ", path.resolve())

    frames = frontier_frames(fronts)
    print("\nin-memory tables:", list(frames), "summary rows", len(frames["summary"]))

    summary = pd.read_csv(f"{stem}_summary.csv")
    report = summary[
        [
            "name",
            "quote_set",
            "n_quotes",
            "prob_lo",
            "prob_hi",
            "loss_lo",
            "loss_hi",
            "depth_lo",
            "depth_hi",
            "mode",
            "status",
            "area_ratio",
        ]
    ].copy()
    for column in ("prob_lo", "prob_hi", "depth_lo", "depth_hi"):
        report[column] *= 100.0
    for column in ("loss_lo", "loss_hi"):
        report[column] *= 10_000.0
    readable = Path("outputs/user-guide/results-export_readable.csv")
    report.to_csv(readable, index=False)
    print("\nreadable units (% and bp):")
    print(report.to_string(index=False))

    vertices = pd.read_csv(f"{stem}_vertices.csv")
    print("\nvertex columns", list(vertices.columns), "rows", len(vertices))
    print(vertices.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
```

## Files

### Summary columns used for decisions

`Frontier.summary()` / `*_summary.csv` have a fixed column order:

```text
date, expiry, days, am_settlement, K, gamma, support_cap, quote_set,
n_quotes, n_strikes, forward, forward_source, rate, discount, spot,
mode, method, status, dimension, prob_lo, prob_hi, loss_lo, loss_hi,
depth_lo, depth_hi, area, area_outer, baseline_area, area_ratio,
interior_ok, interior_margin, support_calls, max_gap, solver_gap,
elapsed_seconds
```

Nested output adds `name` at the front of the CSV.

| Field | Question it answers |
|---|---|
| `quote_set` / `n_quotes` | What the information set is |
| `forward_source` | Where the forward came from |
| `mode` / `method` / `status` | Hard versus relaxed, how it was reconstructed, how far it was certified |
| `prob_*` / `loss_*` / `depth_*` | Marginal intervals of the three quantities (decimals) |
| `area_ratio` | How much tighter the joint constraint is than separate estimates combined afterward |
| `interior_ok` | Whether a further inward shrink of bids and asks remains feasible |
| `solver_gap` | At \(\gamma=0\), largest LP dual gap; at \(\gamma>0\), largest cone violation |

An `area_ratio` well below 1 means one measure has to explain probability and loss together. `p0` changes only the transformed area; it leaves `inner` / `outer` unchanged.

The unknown true set sits between inner and outer. When a coordinate is too narrow, `method=degenerate`, `dimension` is 0 or 1, and area is 0.

At \(\gamma>0\) the depth endpoints can be NaN while the joint set is still complete. Vertex-table `depth=loss/prob` is then a geometric reference, not a certified interval.

### Five export tables

`write_frontier(fronts, path_stem, *, formats=("csv", "json"), certificate=None)` creates parent directories and returns a list of paths.

| File | Question it answers |
|---|---|
| `{stem}_summary.csv` | Interval and status, one row per set |
| `{stem}_vertices.csv` | Inner/outer vertices; columns `kind, order, prob, loss, depth` |
| `{stem}_halfplanes.csv` | Outer set \(a_p p+a_L L\le h\) (raw decimals) |
| `{stem}_quotes.csv` | Quotes that **actually entered the program**, not the raw full cross-section |
| `{stem}_binding.csv` | Only when `certificate` is passed |
| `{stem}.json` | `{"frontiers": {...}, "certificate": null or a list of records}` |

For a single `Frontier`, the JSON key is its `quote_set`. `frontier_frames` returns the same four in-memory tables and writes nothing.

`to_dict()` / `describe()` / `vertices()` / `halfplane_table()` / `depth_boundary()` live on `Frontier`. There is no `analyze` or `export_all`.

Term-structure, threshold, and series tables are saved with `DataFrame.to_csv`. Filter failed rows first:

```python
ok = table[table["error"].fillna("").eq("")].copy()
```

### Units

| Quantity | CSV / `summary` / `scenario_check` | Figure |
|---|---|---|
| Probability | Decimal in \([0,1]\) | % |
| Loss | Decimal relative to the forward | bp by default (×10000). 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1% |
| Depth | Decimal | % |

Keep both the raw-decimal CSV and a readable table with units.

The script writes `outputs/user-guide/results-export_summary.csv` and the other three tables, plus `_binding.csv`, `.json`, and `results-export_readable.csv`.

The readable put-wing row is about probability 0.066%–0.693%, loss 1.12–1.75 bp, depth 1.62%–17.1%.

## Options

| Task | Call |
|---|---|
| JSON only | `formats=("json",)` |
| CSV only | `formats=("csv",)` |
| Single layer | `write_frontier(compute_frontier(cs, 0.90), stem)` |
| No certificate | Omit `certificate=` |
| Physical export | Switch to the 2020 cross-section and the matching `gamma` / `quote_set` / `support_cap` |

## Common errors

- **Using `write_frontier` to save figures.** Figures require `save_figure`.
- **Feeding the readable table back into `contains`.** Percents and bp are two orders of magnitude off.
- **Looking for a `plot` field in the JSON.** There is none. Figures are saved separately.
- **Comparing `*_summary.csv` files with different `support_cap` and no label.** Those are different identification problems.
- **Replacing certified intervals with min/max of vertex `loss/prob` when `depth` is NaN.** The report should say “geometric ratio, uncertified.”
