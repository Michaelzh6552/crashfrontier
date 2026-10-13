from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from crashfrontier.data.parse import parse_dates, parse_numeric
from crashfrontier.data.readers import read_table
from crashfrontier.data.schema import ALIASES, normalize_name
from crashfrontier.errors import SchemaError

__all__ = [
    "SpotTable",
]

_PRICE_ALIASES = (
    "close",
    "price",
    "adj_close",
    "adjclose",
    "spindx",
    "level",
    "last",
    "value",
    "spot",
)


class SpotTable:
    """Exact-date spot lookup; missing dates return NaN."""

    def __init__(self, table: pd.DataFrame, *, dayfirst: bool | None = None) -> None:
        self._dayfirst = dayfirst
        frame = _prepare_spot_table(table, dayfirst=dayfirst)
        self._values = {
            pd.Timestamp(date).normalize(): float(price)
            for date, price in frame[["date", "spot"]].itertuples(index=False, name=None)
        }

    @classmethod
    def from_path(cls, path, *, dayfirst: bool | None = None) -> SpotTable:
        """Read a spot table from disk."""
        return cls(read_table(Path(path)), dayfirst=dayfirst)

    def spot(self, date) -> float:
        """Spot on ``date``, or NaN if that date is absent."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        value = self._values.get(stamp)
        if value is None:
            return float("nan")
        return float(value)


def _prepare_spot_table(table: pd.DataFrame, *, dayfirst: bool | None) -> pd.DataFrame:
    date_col = _find_date_column(table, dayfirst=dayfirst)
    price_col = _require_column(table.columns, _PRICE_ALIASES, "spot")
    frame = pd.DataFrame(
        {
            "date": parse_dates(table[date_col], dayfirst=dayfirst),
            "spot": parse_numeric(table[price_col]),
        }
    )
    frame = frame.dropna(subset=["date", "spot"])
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


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
