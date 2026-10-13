from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["QuoteFilter"]

ROW_RULES = (
    "cp_flag",
    "positive_strike",
    "nonnegative_bid",
    "positive_ask",
    "uncrossed",
    "min_volume",
    "min_open_interest",
    "days_window",
)
NORMALIZED_RULES = (
    "finite_normalized",
    "min_width",
    "one_quote_per_strike",
)
REPORT_RULES = ROW_RULES + NORMALIZED_RULES


@dataclass(frozen=True)
class QuoteFilter:
    """Row-level and post-normalization quote filters with per-rule drop counts."""

    min_volume: float = 1.0
    min_open_interest: float = 0.0
    positive_strike: bool = True
    nonnegative_bid: bool = True
    positive_ask: bool = True
    uncrossed: bool = True
    min_days: int = 7
    max_days: int = 120
    min_width: float = 0.0
    require_finite: bool = True
    cp_flags: tuple[str, ...] = ("C", "P")
    one_quote_per_strike: bool = False

    @classmethod
    def default(cls) -> QuoteFilter:
        """Default filter: volume at least 1, remaining rules at packaged defaults."""
        return cls()

    @classmethod
    def permissive(cls) -> QuoteFilter:
        """Default rules with the volume floor turned off."""
        return cls(min_volume=0.0)

    def apply_rows(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Apply canonical-row filters sequentially and count removals."""
        current = rows.copy()
        report = {name: 0 for name in ROW_RULES}
        volume_missing = _column_all_missing(rows, "volume")
        report["volume_unavailable"] = int(volume_missing)
        current, report["cp_flag"] = _drop(current, current["cp_flag"].isin(self.cp_flags))
        if self.positive_strike:
            current, report["positive_strike"] = _drop(current, current["strike"] > 0.0)
        if self.nonnegative_bid:
            current, report["nonnegative_bid"] = _drop(current, current["bid"] >= 0.0)
        if self.positive_ask:
            current, report["positive_ask"] = _drop(current, current["ask"] > 0.0)
        if self.uncrossed:
            current, report["uncrossed"] = _drop(current, current["ask"] >= current["bid"])
        if self.min_volume > 0.0 and not volume_missing:
            current, report["min_volume"] = _drop(current, current["volume"] >= self.min_volume)
        if self.min_open_interest > 0.0:
            current, report["min_open_interest"] = _drop(
                current, current["open_interest"] >= self.min_open_interest
            )
        current, report["days_window"] = _drop(
            current,
            (current["days"] >= self.min_days) & (current["days"] <= self.max_days),
        )
        report["kept"] = int(len(current))
        return current.reset_index(drop=True), report

    def apply_normalized(self, quotes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Apply post-normalization filters sequentially and count removals."""
        current = quotes.copy()
        report = {name: 0 for name in NORMALIZED_RULES}
        if self.require_finite:
            x_values = current["x"].to_numpy(dtype=float, copy=False)
            lower = current["norm_l"].to_numpy(dtype=float, copy=False)
            upper = current["norm_u"].to_numpy(dtype=float, copy=False)
            finite = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
            current, report["finite_normalized"] = _drop(current, finite)
        if self.min_width > 0.0:
            width = current["norm_u"] - current["norm_l"]
            current, report["min_width"] = _drop(current, width >= self.min_width)
        if self.one_quote_per_strike and len(current):
            before = len(current)
            current = _narrowest_per_strike(current)
            report["one_quote_per_strike"] = int(before - len(current))
        report["kept"] = int(len(current))
        return current.reset_index(drop=True), report


def _drop(frame: pd.DataFrame, mask) -> tuple[pd.DataFrame, int]:
    keep = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    return frame.loc[keep], int((~keep).sum())


def _column_all_missing(frame: pd.DataFrame, name: str) -> bool:
    if name not in frame.columns:
        return True
    return bool(frame[name].isna().all())


def _narrowest_per_strike(quotes: pd.DataFrame) -> pd.DataFrame:
    work = quotes.copy()
    work["_width"] = work["norm_u"] - work["norm_l"]
    work = work.sort_values(["cp_flag", "strike", "_width", "quote_id"], kind="mergesort")
    kept = work.drop_duplicates(subset=["cp_flag", "strike"], keep="first")
    return kept.drop(columns=["_width"])
