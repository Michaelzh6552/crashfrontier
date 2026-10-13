# Report page and common figures

Plotting functions return a `matplotlib.figure.Figure`; `save_figure` writes it to disk. The script below uses the 2017 risk-neutral nest to produce left/right panels, a 2×2 report, a quote figure, and a conditional-depth figure.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `reports.py` and run `python reports.py`.

```python
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import (
    Dataset,
    binding_quotes,
    nested_frontiers,
    plot_binding_quotes,
    plot_conditional_depth,
    plot_frontier,
    plot_frontier_panels,
    plot_quotes,
    plot_report,
    save_figure,
)


def main() -> None:
    output = Path("outputs/user-guide/reports")
    ds = Dataset.from_directory(Path(os.environ["CRASHFRONTIER_DATA"]))
    cs = ds.cross_section("2017-07-05", target_days=14, window=(7, 21))
    fronts = nested_frontiers(
        cs,
        K=0.90,
        gamma=0,
        sets=("put_pair", "local_eight", "put_wing"),
    )
    wing = fronts["put_wing"]
    certificate = binding_quotes(cs, 0.90, direction=(1.0, 0.0), quote_set="put_wing")

    written = []
    written += save_figure(plot_frontier(fronts), output / "frontier", formats=("png",))
    written += save_figure(plot_conditional_depth(fronts), output / "depth", formats=("png",))
    written += save_figure(plot_frontier_panels(fronts), output / "panels", formats=("png", "pdf"))
    written += save_figure(
        plot_quotes(cs, 0.90, used=wing.quotes, highlight=certificate),
        output / "quotes",
        formats=("png",),
    )
    written += save_figure(plot_binding_quotes(certificate), output / "binding", formats=("png",))
    written += save_figure(
        plot_report(cs, fronts, certificate=certificate),
        output / "report",
        formats=("png", "pdf"),
    )
    for path in written:
        print(path.resolve(), path.exists())


if __name__ == "__main__":
    main()
```

## Figures

`fronts` may be a `Frontier`, a name→frontier dictionary, or a sequence. Nested dictionaries are colored `put_pair` → `local_eight` → `put_wing`.

| Function | What it draws |
|---|---|
| `plot_frontier` | Probability–loss joint set. `fill=True` fills the inner set lightly and draws a solid outline; `outline_outer=True` adds a short dashed outer line when that set is distinct. `loss_unit` is `"bp"` or `"percent"`. Default loss axes spell out basis points. 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%. `annotate=True` writes intervals on the figure |
| `plot_conditional_depth` | \(s=L/p\). When probability spans an order of magnitude the horizontal axis may switch to log. `inset=True` may add a full-range inset |
| `plot_frontier_panels` | Left and right panels; the default title includes date, expiry, and \(K\), with the interval caption under it |
| `plot_quotes` | Horizontal axis: normalized strike \(x=\kappa/F\); vertical axis: normalized call price; the vertical dashed line is \(K\). `used=` fades quotes that did not enter that layer; `highlight=` can mark binding quotes. Role colors are separate from the information-set colors |
| `plot_binding_quotes` | Absolute dual weights (log vertical axis). A repeated weight is labeled once |
| `plot_report` | 2×2: quotes, joint set, depth, certificate or text summary |

At \(\gamma>0\) do not pass `binding_quotes`: that function is defined only at \(\gamma=0\). The fourth cell of a physical report page uses a text summary.

`save_figure(fig, path_stem, formats=("pdf", "png"), dpi=300)` creates parent directories. If the stem already ends in `.png`, that suffix is stripped before formats are appended.

Term structure / threshold / time series have `plot_term_structure`, `plot_threshold_profile`, and `plot_time_series` on the later pages. They drop rows whose `error` is non-empty.

The script writes `outputs/user-guide/reports/frontier.png`, `depth.png`, `panels.png` / `.pdf`, `quotes.png`, `binding.png`, and `report.png` / `.pdf`. On the 2017-07-05 slice the default report title is `2017-07-05 → 2017-07-19 (14d), K=0.90`. Lower left is low probability and low loss. Navy, teal, and gold are the put pair, local eight, and put wing.

<figure class="cf-example-figure">
  <img src="../assets/images/examples/05_full_report.png" alt="Full report with option quotes, joint frontier, conditional depth, and dual certificate" loading="lazy">
  <figcaption>
    2017-07-05 → 2017-07-19 (14 days), K=0.90, γ=0. Quote cross-section, nested joint set, conditional depth, and the dual certificate.
  </figcaption>
</figure>

## Options

| Task | Call |
|---|---|
| Loss in percent | `plot_frontier(fronts, loss_unit="percent")` |
| No outer outline | `plot_frontier(fronts, outline_outer=False)` |
| Force a linear depth axis | `plot_conditional_depth(fronts, log_p=False)` |
| Custom title | `plot_frontier_panels(fronts, title="My title")` |
| PNG only | `formats=("png",)` |
| Physical single-layer figure | Switch to the 2020 cross-section and `compute_frontier(..., gamma=2, quote_set="put_wing", support_cap=5.0)` |

## Common errors

- **Calling `frontier.plot()`.** It does not exist. Use the `plot_*` functions above.
- **Expecting `plot_*` to write files.** You must call `save_figure`.
- **Omitting `cs` from `plot_report`.** The signature is `(cs, fronts, *, certificate=None, title=None)`.
- **Computing `binding_quotes` on a physical frontier.** `ValueError: binding_quotes is defined for gamma=0`.
- **Writing the stem as `out.png` and also requesting PDF.** `save_figure` strips the suffix and then adds `.png` / `.pdf` separately; that is usually fine. Do not append another extension yourself.
