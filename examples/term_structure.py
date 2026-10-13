from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from crashfrontier import Dataset, plot_term_structure, plot_threshold_profile, save_figure
from crashfrontier.series import term_structure, threshold_profile

# 2017-07-05 term structure and 14-day threshold profile.
# CRASHFRONTIER_DATA must contain that date.
DATE = "2017-07-05"
TENORS = (7, 14, 30, 60, 90)
K = 0.90
GAMMA = 0
QUOTE_SET = None
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
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
    kwargs: dict = {}
    if QUOTE_SET is not None:
        kwargs["quote_set"] = QUOTE_SET
    table, _fronts = term_structure(
        ds, DATE, tenors=TENORS, K=K, gamma=GAMMA, **kwargs
    )
    print(table.to_string(index=False))
    save_figure(plot_term_structure(table), f"{OUT_DIR}/term_structure")

    cs = ds.cross_section(DATE, target_days=14, window=(7, 21))
    profile, _ = threshold_profile(cs, THRESHOLDS, gamma=GAMMA, **kwargs)
    print(profile.to_string(index=False))
    save_figure(plot_threshold_profile(profile), f"{OUT_DIR}/threshold_profile")


if __name__ == "__main__":
    main()
