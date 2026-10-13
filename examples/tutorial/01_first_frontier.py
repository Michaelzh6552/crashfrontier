from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import Dataset, compute_frontier, plot_frontier_panels, save_figure, write_frontier

# 2017-07-05, target 14 days, K=0.90, default put wing.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TARGET_DAYS = 14
WINDOW = (7, 21)
K = 0.90
OUT_STEM = Path("outputs/tutorial/01_first_frontier")


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
    info = ds.describe()
    print("date_format", info["date_format"], "strike_scale", info["strike_scale"])

    expiries = ds.expiries(DATE)
    print(expiries.to_string(index=False))

    cs = ds.cross_section(DATE, target_days=TARGET_DAYS, window=WINDOW)
    print(cs.summary())

    frontier = compute_frontier(cs, K)
    print(frontier.describe())

    written = []
    fig = plot_frontier_panels(frontier)
    written.extend(save_figure(fig, OUT_STEM))
    plt.close(fig)
    written.extend(write_frontier(frontier, OUT_STEM))
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
