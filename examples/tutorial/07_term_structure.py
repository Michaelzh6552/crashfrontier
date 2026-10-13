from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import Dataset, plot_term_structure, save_figure, term_structure

# 2017-07-05, gamma=0, target tenors 7, 14, and 30.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TENORS = (7, 14, 30)
K = 0.90
OUT_STEM = Path("outputs/tutorial/07_term_structure")


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
    table, fronts = term_structure(ds, DATE, tenors=TENORS, K=K, gamma=0)
    print(table.to_string(index=False))
    print("computed tenors", sorted(fronts))

    csv_path = Path(f"{OUT_STEM}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    fig = plot_term_structure(table)
    written = save_figure(fig, OUT_STEM)
    plt.close(fig)
    for path in [*written, csv_path]:
        print("wrote", path)


if __name__ == "__main__":
    main()
