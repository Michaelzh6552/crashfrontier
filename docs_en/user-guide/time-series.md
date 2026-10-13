# Time series: several observation dates

`frontier_series` repeats the same specification across observation dates. The script samples 2017 Q1 weekly and, on each date, selects options with about 30 days to expiry. The short run computes two weeks; a full-quarter call follows.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick dates from `ds.available_dates()`. A full quarter is much slower; the default script runs only the first two sampled weeks.

Save as `time_series_run.py` and run `python time_series_run.py`.

```python
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import Dataset, frontier_series, plot_time_series, save_figure
from crashfrontier.series import pick_dates


def main() -> None:
    output = Path("outputs/user-guide/time-series")
    output.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_directory(Path(os.environ["CRASHFRONTIER_DATA"]))

    picked = pick_dates(
        ds.available_dates(),
        start="2017-01-01",
        end="2017-03-31",
        freq="W",
    )
    print("weekly sample (Wednesday first, then Tue, Thu, Mon, Fri):")
    print([stamp.date() for stamp in picked])

    # Short run: first two weeks to confirm the pipeline. Use dates=picked for the full quarter.
    table = frontier_series(
        ds,
        dates=picked[:2],
        K=0.90,
        gamma=0,
        target_days=30,
        window=(20, 45),
        snap="backward",
        on_error="skip",
        workers=1,
    )
    cols = [
        "requested_date",
        "date",
        "expiry",
        "days",
        "am_settlement",
        "prob_lo",
        "prob_hi",
        "loss_lo",
        "loss_hi",
        "depth_lo",
        "depth_hi",
        "error",
    ]
    print(table[cols].to_string(index=False))

    csv_path = Path("outputs/user-guide/time-series.csv")
    table.to_csv(csv_path, index=False)
    for value in ("prob", "loss", "depth"):
        fig = plot_time_series(table, value=value)
        save_figure(fig, Path(f"{output}_{value}"), formats=("png", "pdf"))
    print("wrote", csv_path.resolve())


if __name__ == "__main__":
    main()
```

For the full 2017Q1 (about 13 weekly points if those dates exist), replace `dates=picked[:2]` with either of:

```python
table = frontier_series(
    ds,
    start="2017-01-01",
    end="2017-03-31",
    freq="W",
    K=0.90,
    gamma=0,
    target_days=30,
    window=(20, 45),
    snap="backward",
    on_error="skip",
    workers=1,
)
```

`workers>1` calls `ds.spec()` and rebuilds the dataset in child processes; in-memory tables fall back to a single process. File-backed sources can use `workers=2`. `cache_path="outputs/user-guide/time-series-cache.csv"` is a result checkpoint, not the option date-index directory.

`cache_path` and CLI `--resume` reuse a successful row only when its signature matches the inputs that change the result, including the requested date, \(K\), `gamma`, quote set, support cap, and cross-section selection. A row with a non-empty `error` is recomputed. A cache file that has no signature is recomputed rather than reused.

## How to read the result

```text
frontier_series(ds, dates=None, *, start=None, end=None, freq="W", K=0.9, gamma=0,
                quote_set=None, days=None, target_days=30, window=(20, 45),
                am_settlement=None, snap="backward", workers=1,
                cache_path=None, on_error="skip", **frontier_kwargs)
```

- Choose either `dates` or `start/end/freq`. Without `dates`, `pick_dates` is used.
- `freq`: `D` / `W` / `M` / `Q`. Weekly sampling prefers Wednesday, then Tuesday, Thursday, Monday, Friday. Month and quarter take the last date with data in the period.
- `snap`: `backward`, `forward`, or `None`. If the requested date is missing from the option source, the default snaps to the past; `None` raises `NoSuchDateError`.
- Do not pass `days` and `target_days` together: once `days` is set, the target window is unused.
- `on_error="skip"` writes the failure into `error` and continues; `"raise"` stops immediately.
- Extra columns: `requested_date`, `error`.

Do **not** add `am_settlement=0` to “align” an older table. The contract nearest 30 days is sometimes SPX morning settlement; forcing the afternoon session changes the sample or fails that date.

`plot_time_series(table, value="prob"|"loss"|"depth")` puts dates on the horizontal axis. Probability and depth are %. Loss is in basis points (bp): 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%. The loss axis spells out “basis points”. Points with a non-empty `error` or NaN depth drop out of that band. The band is light and the midline is the darker line. Optional `overlay=` stacks a series on the right axis.

The first two weekly points from `pick_dates` in 2017Q1 are `2017-01-04` and `2017-01-11` (Wednesday preferred); both days are 30-day afternoon sessions:

| requested_date | days | am_settlement | prob_lo | prob_hi |
|---|---:|---:|---:|---:|
| 2017-01-04 | 30 | 0 | 0.010004 | 0.016505 |
| 2017-01-11 | 30 | 0 | 0.006253 | 0.015069 |

The script writes `outputs/user-guide/time-series.csv` and `time-series_prob` / `_loss` / `_depth` PNG and PDF. The full quarter runs from about `2017-01-04` to `2017-03-29`, 13 weekly points; a few days may have `am_settlement=1`.

<div class="cf-example-gallery">
  <figure class="cf-example-figure">
    <img src="../assets/images/examples/09_time_series_prob.png" alt="Crash-probability time-series bounds" loading="lazy">
    <figcaption>Weekly γ=0, K=0.90, about 30 days (window 20–45), 2017-01-04 through 2017-01-18. The two-row table above is 2017-01-04 and 2017-01-11 only.</figcaption>
  </figure>
  <figure class="cf-example-figure">
    <img src="../assets/images/examples/09_time_series_loss.png" alt="Expected crash-loss time-series bounds" loading="lazy">
    <figcaption>Expected loss in basis points on the same 2017-01-04–2017-01-18 weekly sample. 1 bp = 0.01%.</figcaption>
  </figure>
  <figure class="cf-example-figure">
    <img src="../assets/images/examples/09_time_series_depth.png" alt="Conditional-depth time-series bounds" loading="lazy">
    <figcaption>Conditional depth (%) on the same 2017-01-04–2017-01-18 weekly sample.</figcaption>
  </figure>
</div>

## Options

| Task | Call |
|---|---|
| Every trading day | `freq="D"` |
| Last date with data in each month | `freq="M"` |
| Pin calendar days to expiry | `days=30` without `window` |
| Physical series | `gamma=2, quote_set="put_wing", support_cap=5.0` (much slower) |
| Resume from a checkpoint | `cache_path="outputs/user-guide/series_cache.csv"` |
| Stop on failure | `on_error="raise"` |
| Overlay spot on the right axis | Build a date-indexed `Series`, then `plot_time_series(..., overlay=spot, overlay_label="Spot")` |

## Common errors

- **Expecting one call to write three figures.** In Python, loop over `value`.
- **`workers>1` with `Dataset.from_frames`.** The object cannot be serialized and falls back to 1.
- **Treating `cache_path` as the option date-index directory.** `cache_path` is a result CSV. The date-byte index of a large option file is a separate `{cache_dir}/{stem}.index.json` (a path hash is added when one dataset opens several files with the same stem).
- **Sunday request date with `snap=None`.** Immediate `NoSuchDateError`.
- **A completely blank depth figure.** Check whether `depth_lo` is all NaN in the CSV, and whether `error` is non-empty.
