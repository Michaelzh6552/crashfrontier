from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from crashfrontier.data.parse import parse_dates, parse_numeric
from crashfrontier.data.readers import read_table
from crashfrontier.data.schema import ALIASES, normalize_name
from crashfrontier.errors import SchemaError

__all__ = [
    "ConstantRate",
    "RateSource",
    "SeriesRate",
    "ZeroCurveRate",
    "ZeroRate",
    "open_rate_source",
]

logger = logging.getLogger("crashfrontier")

_RATE_ALIASES = ("rate", "zero_rate", "yield", "zero")
_PERCENT = 100.0
_UNITS = frozenset({"percent", "decimal"})


@runtime_checkable
class RateSource(Protocol):
    name: str

    def rate(self, date: pd.Timestamp, days: int) -> float: ...


class SeriesRate:
    """Single-tenor rate series, forward-filled by observation date."""

    name = "series"

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        unit: str = "percent",
        dayfirst: bool | None = None,
    ) -> None:
        if unit not in _UNITS:
            raise SchemaError(f"rate unit must be 'percent' or 'decimal', got {unit!r}")
        self._dayfirst = dayfirst
        frame = _prepare_series_table(table, unit=unit, dayfirst=dayfirst)
        self._dates = pd.DatetimeIndex(frame["date"])
        self._values = frame["rate"].to_numpy(dtype=float)

    def rate(self, date: pd.Timestamp, days: int) -> float:
        """Last observation on or before ``date``, as a decimal rate."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        position = int(self._dates.searchsorted(stamp, side="right")) - 1
        if position < 0:
            raise ValueError(
                f"no rate observation on or before {stamp.date()}; pass rates=0.0 or a constant"
            )
        return float(self._values[position])


class ZeroCurveRate:
    """OptionMetrics zero curve with linear interpolation in days."""

    name = "zero_curve"

    def __init__(self, table: pd.DataFrame, *, dayfirst: bool | None = None) -> None:
        self._dayfirst = dayfirst
        self._curves = _prepare_zero_curves(table, dayfirst=dayfirst)
        self._curve_dates = sorted(self._curves)

    def rate(self, date: pd.Timestamp, days: int) -> float:
        """Interpolate the latest curve on or before ``date``; endpoints are held."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        eligible = [curve_date for curve_date in self._curve_dates if curve_date <= stamp]
        if not eligible:
            raise ValueError(
                f"no zero curve on or before {stamp.date()}; pass rates=0.0 or a constant"
            )
        nodes, values = self._curves[eligible[-1]]
        return float(np.interp(float(days), nodes, values))


class ConstantRate:
    """Constant decimal rate for every date and tenor."""

    name = "constant"

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def rate(self, date: pd.Timestamp, days: int) -> float:
        return self._value


class ZeroRate:
    """Zero rate; logs a warning on the first call."""

    name = "zero"

    def __init__(self) -> None:
        self._warned = False

    def rate(self, date: pd.Timestamp, days: int) -> float:
        if not self._warned:
            logger.warning("no interest-rate data; using r = 0")
            self._warned = True
        return 0.0


def open_rate_source(rates, *, dayfirst: bool | None = None, unit: str = "percent") -> RateSource:
    """Build a rate source from None, a constant, a path, or a DataFrame."""
    if rates is None:
        return ZeroRate()
    if isinstance(rates, bool):
        raise SchemaError("boolean is not a rate source")
    if isinstance(rates, (int, float)):
        return ConstantRate(float(rates))
    if isinstance(rates, (str, Path)):
        return _source_from_frame(read_table(Path(rates)), dayfirst=dayfirst, unit=unit)
    if isinstance(rates, pd.DataFrame):
        return _source_from_frame(rates, dayfirst=dayfirst, unit=unit)
    raise SchemaError(f"unrecognized rate source {type(rates)!r}")


def _source_from_frame(
    frame: pd.DataFrame,
    *,
    dayfirst: bool | None,
    unit: str,
) -> RateSource:
    if any(normalize_name(column) == "days" for column in frame.columns):
        return ZeroCurveRate(frame, dayfirst=dayfirst)
    return SeriesRate(frame, unit=unit, dayfirst=dayfirst)


def _prepare_series_table(
    table: pd.DataFrame,
    *,
    unit: str,
    dayfirst: bool | None,
) -> pd.DataFrame:
    date_col = _find_date_column(table, dayfirst=dayfirst)
    value_col = _first_numeric_column(table, exclude={date_col})
    frame = pd.DataFrame(
        {
            "date": parse_dates(table[date_col], dayfirst=dayfirst),
            "rate": parse_numeric(table[value_col]),
        }
    )
    if unit == "percent":
        frame["rate"] = frame["rate"] / _PERCENT
    frame = frame.sort_values("date", kind="mergesort")
    frame["rate"] = frame["rate"].ffill()
    frame = frame.dropna(subset=["date", "rate"])
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


def _prepare_zero_curves(
    table: pd.DataFrame,
    *,
    dayfirst: bool | None,
) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    date_col = _find_date_column(table, dayfirst=dayfirst)
    days_col = _require_column(table.columns, ("days",), "days")
    rate_col = _require_column(table.columns, _RATE_ALIASES, "rate")
    frame = pd.DataFrame(
        {
            "date": parse_dates(table[date_col], dayfirst=dayfirst),
            "days": parse_numeric(table[days_col]),
            "rate": parse_numeric(table[rate_col]) / _PERCENT,
        }
    )
    frame = frame.dropna(subset=["date", "days", "rate"])
    frame = frame.drop_duplicates(subset=["date", "days"], keep="last")
    curves: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for curve_date, group in frame.groupby("date", sort=True):
        ordered = group.sort_values("days", kind="mergesort")
        nodes = ordered["days"].to_numpy(dtype=float)
        values = ordered["rate"].to_numpy(dtype=float)
        if len(nodes) == 0:
            continue
        curves[pd.Timestamp(curve_date).normalize()] = (nodes, values)
    if not curves:
        raise SchemaError("zero curve table has no valid date/days/rate rows")
    return curves


def _find_date_column(frame: pd.DataFrame, *, dayfirst: bool | None) -> str:
    matched = _match_column(frame.columns, ALIASES["date"])
    if matched is not None:
        return matched
    for column in frame.columns:
        try:
            parsed = parse_dates(frame[column], dayfirst=dayfirst)
        except (SchemaError, ValueError, TypeError):
            continue
        if parsed.notna().any():
            return str(column)
    raise SchemaError(
        f"missing date column; aliases {list(ALIASES['date'])}; columns {list(frame.columns)}"
    )


def _first_numeric_column(frame: pd.DataFrame, *, exclude: set[str]) -> str:
    for column in frame.columns:
        if column in exclude:
            continue
        parsed = parse_numeric(frame[column])
        if parsed.notna().any():
            return str(column)
    raise SchemaError(f"no numeric value column; columns {list(frame.columns)}")


def _require_column(columns: Sequence[str], aliases: Sequence[str], field: str) -> str:
    match = _match_column(columns, aliases)
    if match is None:
        raise SchemaError(
            f"missing {field} column; aliases {list(aliases)}; columns {list(columns)}"
        )
    return match


def _match_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    by_normalized: dict[str, str] = {}
    for column in columns:
        by_normalized.setdefault(normalize_name(column), str(column))
    for alias in aliases:
        original = by_normalized.get(normalize_name(alias))
        if original is not None:
            return original
    return None


def _as_timestamp(value, *, dayfirst: bool | None = None) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize(None) if value.tzinfo is not None else value
        return stamp.normalize()
    parsed = parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0]
    if pd.isna(parsed):
        raise SchemaError(f"could not parse date {value!r}")
    return pd.Timestamp(parsed).normalize()
