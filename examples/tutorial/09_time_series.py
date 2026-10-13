from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import Dataset, frontier_series, plot_time_series, save_figure

# Weekly gamma=0 series, 2017-01-01 through 2017-01-18, target about 30 days.
# CRASHFRONTIER_DATA must contain that window.
START = "2017-01-01"
END = "2017-01-18"
FREQ = "W"
TARGET_DAYS = 30
WINDOW = (20, 45)
K = 0.90
OUT_STEM = Path("outputs/tutorial/09_time_series")


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
    table = frontier_series(
        ds,
        start=START,
        end=END,
        freq=FREQ,
        K=K,
        gamma=0,
        target_days=TARGET_DAYS,
        window=WINDOW,
        snap="backward",
        workers=1,
        on_error="skip",
    )
    print(table.to_string(index=False))

    csv_path = Path(f"{OUT_STEM}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    written = [csv_path]
    for value in ("prob", "loss", "depth"):
        fig = plot_time_series(table, value=value)
        written.extend(save_figure(fig, Path(f"{OUT_STEM}_{value}")))
        plt.close(fig)
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
