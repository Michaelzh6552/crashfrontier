from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import Dataset, plot_time_series, save_figure
from crashfrontier.series import frontier_series

# Weekly series, 2017 Q1, target about 30 days.
# CRASHFRONTIER_DATA must contain that window.
START = "2017-01-01"
END = "2017-03-31"
FREQ = "W"
TARGET = 30
WINDOW = (20, 45)
K = 0.90
GAMMA = 0
QUOTE_SET = None
WORKERS = 2
OUT_DIR = "outputs"


def _data_root() -> Path:
    raw = os.environ.get("CRASHFRONTIER_DATA")
    if not raw:
        raise SystemExit("Set CRASHFRONTIER_DATA to the directory that contains your option tables.")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise SystemExit(f"CRASHFRONTIER_DATA is not a directory: {root}")
    return root


def main() -> None:
    ds = Dataset.from_directory(_data_root())
    kwargs: dict = {
        "start": START,
        "end": END,
        "freq": FREQ,
        "K": K,
        "gamma": GAMMA,
        "target_days": TARGET,
        "window": WINDOW,
        "workers": WORKERS,
    }
    if QUOTE_SET is not None:
        kwargs["quote_set"] = QUOTE_SET
    table = frontier_series(ds, **kwargs)
    print(table.to_string(index=False))
    stem = f"{OUT_DIR}/time_series"
    table.to_csv(f"{stem}.csv", index=False)
    for value in ("prob", "loss", "depth"):
        fig = plot_time_series(table, value=value)
        save_figure(fig, f"{stem}_{value}")


if __name__ == "__main__":
    main()
