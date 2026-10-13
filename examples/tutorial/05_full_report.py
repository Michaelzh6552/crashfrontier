from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import (
    Dataset,
    binding_quotes,
    nested_frontiers,
    plot_report,
    save_figure,
    write_frontier,
)

# 2017-07-05, target 14 days, nested report and gamma=0 certificate.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TARGET_DAYS = 14
WINDOW = (7, 21)
K = 0.90
SETS = ("put_pair", "local_eight", "put_wing")
OUT_STEM = Path("outputs/tutorial/05_full_report")


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
    fronts = nested_frontiers(cs, K, sets=SETS, gamma=0)
    certificate = binding_quotes(cs, K, direction=(1.0, 0.0))

    print(fronts["put_wing"].describe())
    print(certificate.to_frame().head().to_string(index=False))

    written = []
    fig = plot_report(cs, fronts, certificate=certificate)
    written.extend(save_figure(fig, OUT_STEM))
    plt.close(fig)
    written.extend(write_frontier(fronts, OUT_STEM, certificate=certificate))
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
