from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.special import ndtr

from crashfrontier import (
    CrossSection,
    Dataset,
    compute_frontier,
    nested_frontiers,
    plot_frontier_panels,
    save_figure,
)

DATE = "2024-01-17"
EXPIRY = "2024-02-16"
FORWARD = 4790.5
RATE = 0.053
SPOT = 4785.0
SIGMA = 0.18
K = 0.90
OUT_DIR = "outputs"


def _black76(forward: float, strike: float, tau: float, sigma: float, discount: float, cp_flag: str) -> float:
    vol = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol
    d2 = d1 - vol
    if cp_flag == "C":
        return discount * (forward * float(ndtr(d1)) - strike * float(ndtr(d2)))
    return discount * (strike * float(ndtr(-d2)) - forward * float(ndtr(-d1)))


def _synthetic_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date = pd.Timestamp(DATE)
    expiry = pd.Timestamp(EXPIRY)
    days = int((expiry - date).days)
    tau = days / 365.0
    discount = math.exp(-RATE * tau)
    half = 0.0005 * FORWARD
    strikes = np.round(np.linspace(0.70 * FORWARD, 1.20 * FORWARD, 25) / 5.0) * 5.0
    rows: list[dict[str, object]] = []
    option_id = 1
    for strike in strikes:
        for cp_flag in ("C", "P"):
            mid = _black76(FORWARD, float(strike), tau, SIGMA, discount, cp_flag)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "exdate": expiry.strftime("%Y-%m-%d"),
                    "cp_flag": cp_flag,
                    "strike_price": float(strike),
                    "best_bid": max(mid - half, 0.0),
                    "best_offer": mid + half,
                    "volume": 10,
                    "optionid": option_id,
                    "am_settlement": 0,
                    "symbol": "SYN",
                    "root": "SYN",
                }
            )
            option_id += 1
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
    options, forwards, rates = _synthetic_market()
    quote_sheet = options.rename(
        columns={
            "exdate": "expiry",
            "strike_price": "strike",
            "best_bid": "bid",
            "best_offer": "ask",
        }
    )[["strike", "cp_flag", "bid", "ask", "volume"]]
    cs_quotes = CrossSection.from_quotes(
        quote_sheet,
        date=DATE,
        expiry=EXPIRY,
        forward=FORWARD,
        rate=RATE,
        spot=SPOT,
    )
    from_quotes = compute_frontier(cs_quotes, K)
    print("from_quotes")
    print(from_quotes.describe())
    print()

    ds = Dataset.from_frames(
        options,
        forwards=forwards,
        rates=rates,
        strike_scale=1.0,
    )
    cs_frames = ds.cross_section(DATE, days=30, am_settlement=0)
    from_frames = nested_frontiers(cs_frames, K)
    print("from_frames nested put_wing")
    print(from_frames["put_wing"].describe())
    print()

    physical = compute_frontier(
        cs_frames, K, gamma=2, quote_set="put_wing", support_cap=5.0
    )
    print("from_frames gamma=2 put_wing")
    print(physical.describe())

    fig = plot_frontier_panels(from_frames)
    save_figure(fig, f"{OUT_DIR}/custom_data_panels")


if __name__ == "__main__":
    main()
