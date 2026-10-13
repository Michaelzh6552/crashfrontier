from __future__ import annotations

import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crashfrontier import (
    ColumnMap,
    Dataset,
    compute_frontier,
    plot_frontier_panels,
    save_figure,
    write_frontier,
)

# Nonstandard column names written to a temporary directory and deleted after the run.
DATE = "2024-01-17"
EXPIRY = "2024-02-16"
FORWARD = 4790.5
RATE = 0.053
SIGMA = 0.18
K = 0.90
OUT_STEM = Path("outputs/tutorial/11_custom_files")


def _ncdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black76(forward: float, strike: float, tau: float, sigma: float, discount: float, cp_flag: str) -> float:
    vol = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol
    d2 = d1 - vol
    if cp_flag == "C":
        return discount * (forward * _ncdf(d1) - strike * _ncdf(d2))
    return discount * (strike * _ncdf(-d2) - forward * _ncdf(-d1))


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date = pd.Timestamp(DATE)
    expiry = pd.Timestamp(EXPIRY)
    tau = (expiry - date).days / 365.0
    discount = math.exp(-RATE * tau)
    half = 0.0005 * FORWARD
    strikes = np.round(np.linspace(0.70 * FORWARD, 1.20 * FORWARD, 21) / 5.0) * 5.0
    rows: list[dict[str, object]] = []
    for strike in strikes:
        for cp_flag in ("C", "P"):
            mid = _black76(FORWARD, float(strike), tau, SIGMA, discount, cp_flag)
            rows.append(
                {
                    "asof_day": date.strftime("%Y-%m-%d"),
                    "mature_on": expiry.strftime("%Y-%m-%d"),
                    "side": cp_flag,
                    "k_pts": float(strike),
                    "px_bid": max(mid - half, 0.0),
                    "px_ask": mid + half,
                    "volume": 10,
                    "am_settlement": 0,
                    "root": "SYN",
                }
            )
    options = pd.DataFrame(rows)
    forwards = pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d")],
            "expiration": [expiry.strftime("%Y-%m-%d")],
            "AMSettlement": [0],
            "ForwardPrice": [FORWARD],
        }
    )
    rates = pd.DataFrame(
        {
            "observation_date": [date.strftime("%Y-%m-%d")],
            "DGS1MO": [RATE * 100.0],
        }
    )
    return options, forwards, rates


def main() -> None:
    options, forwards, rates = _synthetic_tables()
    column_map = ColumnMap.detect(
        options.columns,
        overrides={
            "date": "asof_day",
            "expiry": "mature_on",
            "cp_flag": "side",
            "strike": "k_pts",
            "bid": "px_bid",
            "ask": "px_ask",
        },
    )

    with tempfile.TemporaryDirectory(prefix="crashfrontier_tutorial_11_") as tmp:
        tmpdir = Path(tmp)
        options_path = tmpdir / "options.csv"
        forwards_path = tmpdir / "forwards.csv"
        rates_path = tmpdir / "rates.csv"
        options.to_csv(options_path, index=False)
        forwards.to_csv(forwards_path, index=False)
        rates.to_csv(rates_path, index=False)

        # Keep the date index inside the temporary directory.
        ds = Dataset(
            options_path,
            forwards=forwards_path,
            rates=rates_path,
            column_map=column_map,
            strike_scale=1.0,
            cache_dir=tmpdir / "CACHE",
        )
        print(ds.describe())
        cs = ds.cross_section(DATE, days=30, am_settlement=0)
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
