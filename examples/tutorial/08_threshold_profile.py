from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import Dataset, plot_threshold_profile, save_figure, threshold_profile

# 2017-07-05, 14-day cross-section, thresholds 0.80 to 0.95.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TARGET_DAYS = 14
WINDOW = (7, 21)
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
OUT_STEM = Path("outputs/tutorial/08_threshold_profile")


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
    table, fronts = threshold_profile(cs, THRESHOLDS, gamma=0)
    print(table.to_string(index=False))
    print("computed K", sorted(fronts))

    csv_path = Path(f"{OUT_STEM}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    fig = plot_threshold_profile(table)
    written = save_figure(fig, OUT_STEM)
    plt.close(fig)
    for path in [*written, csv_path]:
        print("wrote", path)


if __name__ == "__main__":
    main()
