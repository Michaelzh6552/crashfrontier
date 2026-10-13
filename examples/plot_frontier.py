from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import (
    Dataset,
    binding_quotes,
    nested_frontiers,
    plot_frontier_panels,
    plot_report,
    save_figure,
)

# 2017-07-05, target 14 days. CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TARGET_DAYS = 14
WINDOW = (7, 21)
K = 0.90
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
    cs = ds.cross_section(DATE, target_days=TARGET_DAYS, window=WINDOW)
    fronts = nested_frontiers(cs, K)
    panels = plot_frontier_panels(fronts)
    save_figure(panels, f"{OUT_DIR}/plot_frontier_panels")
    certificate = binding_quotes(cs, K, direction=(1.0, 0.0))
    report = plot_report(cs, fronts, certificate=certificate)
    save_figure(report, f"{OUT_DIR}/plot_frontier_report")
    print(fronts["put_wing"].describe())


if __name__ == "__main__":
    main()
