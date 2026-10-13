from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from crashfrontier import (
    Dataset,
    compute_frontier,
    plot_frontier_panels,
    portfolio_bound,
    save_figure,
    scenario_check,
    write_frontier,
)

# 2017-07-05, target 14 days, gamma=0 scenario checks.
# Probabilities and losses are table decimals.
# CRASHFRONTIER_DATA must contain that slice.
DATE = "2017-07-05"
TARGET_DAYS = 14
WINDOW = (7, 21)
K = 0.90
OUTSIDE = (0.02, 0.004)
OUT_STEM = Path("outputs/tutorial/06_scenario_check")


def _data_root() -> Path:
    raw = os.environ.get("CRASHFRONTIER_DATA")
    if not raw:
        raise SystemExit("Set CRASHFRONTIER_DATA to the directory that contains your option tables.")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise SystemExit(f"CRASHFRONTIER_DATA is not a directory: {root}")
    return root


def _record(label: str, result: dict) -> dict[str, object]:
    return {
        "label": label,
        "feasible": bool(result["feasible"]),
        "geometry": result["geometry"],
        "method": result["method"],
        "contains": result["frontier"].contains(result["_prob"], result["_loss"]),
        "prob": result["_prob"],
        "loss": result["_loss"],
    }


def main() -> None:
    ds = Dataset.from_directory(_data_root())
    cs = ds.cross_section(DATE, target_days=TARGET_DAYS, window=WINDOW)
    frontier = compute_frontier(cs, K)

    inner = frontier.vertices()
    inner = inner.loc[inner["kind"].eq("inner")]
    inside_prob = float(inner["prob"].mean())
    inside_loss = float(inner["loss"].mean())

    checks = []
    for label, prob, loss in (
        ("inner_centroid", inside_prob, inside_loss),
        ("tutorial_stress", OUTSIDE[0], OUTSIDE[1]),
    ):
        result = scenario_check(cs, K, prob, loss, gamma=0, quote_set="put_wing")
        result["_prob"] = prob
        result["_loss"] = loss
        row = _record(label, result)
        checks.append(row)
        print(label, {key: row[key] for key in ("feasible", "geometry", "method", "contains")})

    price_lo, price_hi = portfolio_bound(cs, K, 1.0, 0.0)
    print("portfolio_bound 1{X<=K}", price_lo, price_hi)

    written = []
    fig = plot_frontier_panels(frontier)
    written.extend(save_figure(fig, OUT_STEM))
    plt.close(fig)
    written.extend(write_frontier(frontier, OUT_STEM))
    payload = {
        "checks": checks,
        "portfolio_bound": {"u_prob": 1.0, "u_loss": 0.0, "lo": price_lo, "hi": price_hi},
    }
    checks_path = Path(f"{OUT_STEM}_checks.json")
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written.append(checks_path)
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
