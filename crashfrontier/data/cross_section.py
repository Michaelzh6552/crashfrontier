from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crashfrontier.data.filters import NORMALIZED_RULES, ROW_RULES, QuoteFilter
from crashfrontier.data.parse import parse_dates
from crashfrontier.data.schema import ALIASES, ColumnMap, canonicalize, normalize_name

__all__ = ["CrossSection", "build_cross_section"]

QUOTE_COLUMNS = (
    "quote_id",
    "symbol",
    "root",
    "cp_flag",
    "strike",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "impl_volatility",
    "x",
    "log_moneyness",
    "norm_bid",
    "norm_ask",
    "norm_l",
    "norm_u",
    "role",
)
_SOURCE_COLUMNS = QUOTE_COLUMNS[:10]
_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class CrossSection:
    """Normalized option quotes for one observation date, expiry, and settlement."""

    date: pd.Timestamp
    expiry: pd.Timestamp
    am_settlement: int
    root: str
    days: int
    tau: float
    forward: float
    forward_source: str
    rate: float
    rate_source: str
    discount: float
    spot: float
    quotes: pd.DataFrame
    filter_report: dict[str, int]
    ambiguous_settlement: bool = False

    @property
    def puts(self) -> pd.DataFrame:
        """Put rows of ``quotes``."""
        return self.quotes.loc[self.quotes["cp_flag"].eq("P")]

    @property
    def calls(self) -> pd.DataFrame:
        """Call rows of ``quotes``."""
        return self.quotes.loc[self.quotes["cp_flag"].eq("C")]

    def summary(self) -> dict[str, object]:
        """Scalar identifiers plus quote counts."""
        puts = self.puts
        calls = self.calls
        return {
            "date": self.date,
            "expiry": self.expiry,
            "am_settlement": self.am_settlement,
            "root": self.root,
            "days": self.days,
            "tau": self.tau,
            "forward": self.forward,
            "forward_source": self.forward_source,
            "rate": self.rate,
            "rate_source": self.rate_source,
            "discount": self.discount,
            "spot": self.spot,
            "n_quotes": int(len(self.quotes)),
            "n_puts": int(len(puts)),
            "n_calls": int(len(calls)),
            "n_put_strikes": int(puts["strike"].nunique()) if len(puts) else 0,
            "n_call_strikes": int(calls["strike"].nunique()) if len(calls) else 0,
            "ambiguous_settlement": bool(self.ambiguous_settlement),
        }

    def __repr__(self) -> str:
        date_text = pd.Timestamp(self.date).strftime("%Y-%m-%d")
        expiry_text = pd.Timestamp(self.expiry).strftime("%Y-%m-%d")
        return (
            f"CrossSection(date={date_text}, expiry={expiry_text}, days={self.days}, "
            f"root={self.root}, F={self.forward}, D={self.discount}, "
            f"n_puts={len(self.puts)}, n_calls={len(self.calls)})"
        )

    @classmethod
    def from_quotes(
        cls,
        quotes: pd.DataFrame,
        *,
        date,
        expiry,
        forward: float,
        rate: float = 0.0,
        spot: float = float("nan"),
        am_settlement: int = 0,
        column_map: ColumnMap | None = None,
        strike_scale: float = 1.0,
        dayfirst: bool | None = None,
        cp_values: dict[str, str] | None = None,
        quote_filter: QuoteFilter = QuoteFilter(),
    ) -> CrossSection:
        """Build a cross-section from a user table of quotes for one date and expiry."""
        date_stamp = _as_timestamp(date, dayfirst=dayfirst)
        expiry_stamp = _as_timestamp(expiry, dayfirst=dayfirst)
        frame = quotes.copy()
        mapping = column_map
        if mapping is None:
            frame = _fill_missing_date_expiry(frame, date_stamp, expiry_stamp)
            mapping = ColumnMap.detect(frame.columns)
        else:
            if mapping.date not in frame.columns:
                frame[mapping.date] = date_stamp
            if mapping.expiry not in frame.columns:
                frame[mapping.expiry] = expiry_stamp
        had_settlement = mapping.am_settlement is not None
        canonical = canonicalize(
            frame,
            mapping,
            strike_scale=strike_scale,
            dayfirst=dayfirst,
            cp_values=cp_values,
        )
        settlement = int(am_settlement)
        canonical = canonical.loc[
            canonical["date"].eq(date_stamp) & canonical["expiry"].eq(expiry_stamp)
        ].copy()
        if had_settlement:
            canonical = canonical.loc[canonical["am_settlement"].eq(settlement)].copy()
        else:
            canonical["am_settlement"] = settlement
        canonical = canonical.reset_index(drop=True)
        return build_cross_section(
            canonical,
            date=date_stamp,
            expiry=expiry_stamp,
            am_settlement=settlement,
            forward=float(forward),
            forward_source="user",
            rate=float(rate),
            rate_source="user",
            spot=float(spot),
            quote_filter=quote_filter,
        )


def build_cross_section(
    rows: pd.DataFrame,
    *,
    date,
    expiry,
    am_settlement,
    forward,
    forward_source,
    rate,
    rate_source,
    spot,
    quote_filter: QuoteFilter,
) -> CrossSection:
    """Filter and normalize canonical option rows into a ``CrossSection``."""
    date_stamp = _as_timestamp(date)
    expiry_stamp = _as_timestamp(expiry)
    days = int((expiry_stamp - date_stamp).days)
    tau = days / _DAYS_PER_YEAR
    forward_value = float(forward)
    rate_value = float(rate)
    discount = math.exp(-rate_value * tau)
    filtered, row_report = quote_filter.apply_rows(rows)
    quotes = _normalize(filtered, forward_value, discount)
    quotes, norm_report = quote_filter.apply_normalized(quotes)
    quotes = quotes.sort_values(["strike", "cp_flag", "quote_id"], kind="mergesort")
    quotes = quotes.loc[:, list(QUOTE_COLUMNS)].reset_index(drop=True)
    return CrossSection(
        date=date_stamp,
        expiry=expiry_stamp,
        am_settlement=int(am_settlement),
        root=_section_root(rows),
        days=days,
        tau=tau,
        forward=forward_value,
        forward_source=str(forward_source),
        rate=rate_value,
        rate_source=str(rate_source),
        discount=discount,
        spot=float(spot),
        quotes=quotes,
        filter_report=_merge_reports(row_report, norm_report),
    )


def _normalize(rows: pd.DataFrame, forward: float, discount: float) -> pd.DataFrame:
    quotes = rows.loc[:, list(_SOURCE_COLUMNS)].copy()
    strike = rows["strike"].to_numpy(dtype=float, copy=False)
    bid = rows["bid"].to_numpy(dtype=float, copy=False)
    ask = rows["ask"].to_numpy(dtype=float, copy=False)
    flags = rows["cp_flag"].to_numpy()
    scale = discount * forward
    x_values = strike / forward
    log_moneyness = np.full(x_values.shape, np.nan)
    positive = x_values > 0.0
    log_moneyness[positive] = np.log(x_values[positive])
    norm_bid = bid / scale
    norm_ask = ask / scale
    is_put = flags == "P"
    is_call = flags == "C"
    quotes["x"] = x_values
    quotes["log_moneyness"] = log_moneyness
    quotes["norm_bid"] = norm_bid
    quotes["norm_ask"] = norm_ask
    quotes["norm_l"] = np.where(is_put, norm_bid + 1.0 - x_values, norm_bid)
    quotes["norm_u"] = np.where(is_put, norm_ask + 1.0 - x_values, norm_ask)
    role = np.full(x_values.shape, "itm", dtype=object)
    role[is_put & (x_values <= 1.0)] = "otm_put"
    role[is_call & (x_values >= 1.0)] = "otm_call"
    quotes["role"] = role
    return quotes.loc[:, list(QUOTE_COLUMNS)]


def _merge_reports(row_report: dict[str, int], norm_report: dict[str, int]) -> dict[str, int]:
    report = {name: int(row_report.get(name, 0)) for name in ROW_RULES}
    for name in NORMALIZED_RULES:
        report[name] = int(norm_report.get(name, 0))
    report["volume_unavailable"] = int(row_report.get("volume_unavailable", 0))
    report["kept"] = int(norm_report.get("kept", 0))
    return report


def _section_root(rows: pd.DataFrame) -> str:
    if "root" not in rows.columns or rows.empty:
        return ""
    roots = pd.unique(rows["root"].astype(str))
    if len(roots) == 1:
        return str(roots[0])
    if len(roots) == 0:
        return ""
    return "MIXED"


def _fill_missing_date_expiry(frame: pd.DataFrame, date_stamp, expiry_stamp) -> pd.DataFrame:
    names = {normalize_name(column) for column in frame.columns}
    date_aliases = {normalize_name(alias) for alias in ALIASES["date"]}
    expiry_aliases = {normalize_name(alias) for alias in ALIASES["expiry"]}
    out = frame
    if names.isdisjoint(date_aliases):
        out = out.copy()
        out["date"] = date_stamp
    if names.isdisjoint(expiry_aliases):
        out = out.copy()
        out["expiry"] = expiry_stamp
    return out


def _as_timestamp(value, *, dayfirst: bool | None = None) -> pd.Timestamp:
    return pd.Timestamp(parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0])
