from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import Dataset, compute_frontier, plot_frontier_panels, save_figure, write_frontier

# 2020-03-25, 30-day PM, gamma=2, put_wing, support_cap=5.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2020-03-25"
DAYS = 30
AM_SETTLEMENT = 0
K = 0.90
GAMMA = 2
QUOTE_SET = "put_wing"
SUPPORT_CAP = 5.0
OUT_STEM = Path("outputs/tutorial/04_gamma2")


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
    cs = ds.cross_section(DATE, days=DAYS, am_settlement=AM_SETTLEMENT)
    print(cs.summary())

    frontier = compute_frontier(
        cs,
        K,
        gamma=GAMMA,
        quote_set=QUOTE_SET,
        support_cap=SUPPORT_CAP,
    )
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
