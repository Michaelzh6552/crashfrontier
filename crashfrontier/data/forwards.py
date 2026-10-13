from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from crashfrontier.data.parse import parse_cp_flag, parse_dates, parse_numeric
from crashfrontier.data.readers import read_table
from crashfrontier.data.schema import ALIASES, normalize_name
from crashfrontier.errors import MissingForwardError, SchemaError

__all__ = [
    "ForwardColumn",
    "ForwardSource",
    "ForwardTable",
    "ParityForward",
    "resolve_forward",
]

logger = logging.getLogger("crashfrontier")

_EXPIRY_ALIASES = ("expiration", "exdate", "expiry", "expiration_date", "maturity")
_FORWARD_ALIASES = ("forwardprice", "forward_price", "forward", "fwd", "fwd_price", "f")
_SETTLEMENT_ALIASES = ("amsettlement", "am_settlement", "am_set_flag", "settlement")
_PARITY_NEAREST = 9
_PARITY_MIN_STRIKES = 2


@runtime_checkable
class ForwardSource(Protocol):
    name: str

    def forward(
        self,
        date: pd.Timestamp,
        expiry: pd.Timestamp,
        am_settlement: int | None,
    ) -> float | None: ...


class ForwardTable:
    """Lookup table of forwards keyed by date, expiry, and settlement."""

    name = "table"

    def __init__(self, table: pd.DataFrame, *, dayfirst: bool | None = None) -> None:
        self._dayfirst = dayfirst
        frame = _canonicalize_forward_table(table, dayfirst=dayfirst)
        self._groups = _index_forward_rows(frame)

    @classmethod
    def from_path(cls, path_or_paths, *, dayfirst: bool | None = None) -> ForwardTable:
        """Read one or more forward tables from disk."""
        frames = [read_table(path) for path in _as_paths(path_or_paths)]
        return cls.from_frame(pd.concat(frames, ignore_index=True), dayfirst=dayfirst)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, dayfirst: bool | None = None) -> ForwardTable:
        """Build a forward table from an in-memory frame."""
        return cls(frame, dayfirst=dayfirst)

    def forward(
        self,
        date: pd.Timestamp,
        expiry: pd.Timestamp,
        am_settlement: int | None,
    ) -> float | None:
        """Return the table forward, or None if the pair is absent."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        maturity = _as_timestamp(expiry, dayfirst=self._dayfirst)
        rows = self._groups.get((stamp, maturity))
        if not rows:
            return None
        return _choose_forward(rows, _normalize_settlement(am_settlement), stamp, maturity)

    def prices_on(self, date: pd.Timestamp) -> list[float]:
        """All table forwards on ``date``, across expiries and settlements."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        return [
            price
            for (observed, _expiry), rows in self._groups.items()
            if observed == stamp
            for _settlement, price in rows
        ]


class ForwardColumn:
    """Median of a day's ``forward_price`` column by expiry and settlement."""

    name = "column"

    def __init__(self, rows: pd.DataFrame, *, dayfirst: bool | None = None) -> None:
        self._dayfirst = dayfirst
        self._groups = _index_column_forwards(rows, dayfirst=dayfirst)

    def forward(
        self,
        date: pd.Timestamp,
        expiry: pd.Timestamp,
        am_settlement: int | None,
    ) -> float | None:
        """Return the column forward for ``expiry``, ignoring ``date``."""
        maturity = _as_timestamp(expiry, dayfirst=self._dayfirst)
        rows = self._groups.get(maturity)
        if not rows:
            return None
        return _choose_forward(rows, _normalize_settlement(am_settlement), None, maturity)


class ParityForward:
    """Forward implied by put-call parity on a raw quote panel."""

    name = "parity"

    def __init__(
        self,
        rows: pd.DataFrame,
        discount: float,
        spot: float | None = None,
        *,
        dayfirst: bool | None = None,
    ) -> None:
        self._dayfirst = dayfirst
        self._discount = float(discount)
        self._spot = float("nan") if spot is None else float(spot)
        self._panel = _prepare_parity_rows(rows, dayfirst=dayfirst)

    def forward(
        self,
        date: pd.Timestamp,
        expiry: pd.Timestamp,
        am_settlement: int | None,
    ) -> float | None:
        """Median implied forward of the nine strikes nearest the reference price."""
        panel = _filter_parity_rows(
            self._panel,
            date,
            expiry,
            _normalize_settlement(am_settlement),
            dayfirst=self._dayfirst,
        )
        if panel.empty:
            return None
        return _implied_forward(panel, self._discount, self._spot)


def resolve_forward(
    chain: Sequence[ForwardSource],
    date,
    expiry,
    am_settlement,
) -> tuple[float, str]:
    """Return the first finite forward in ``chain`` and the source name."""
    names = [source.name for source in chain]
    for source in chain:
        value = source.forward(date, expiry, am_settlement)
        if value is not None and bool(np.isfinite(value)):
            return float(value), source.name
    stamp = _as_timestamp(date)
    maturity = _as_timestamp(expiry)
    raise MissingForwardError(
        f"no forward for date={stamp.date()} expiry={maturity.date()} "
        f"am_settlement={am_settlement}; tried {names}"
    )


def _canonicalize_forward_table(
    table: pd.DataFrame,
    *,
    dayfirst: bool | None,
) -> pd.DataFrame:
    date_col = _require_column(table.columns, ALIASES["date"], "date")
    expiry_col = _require_column(table.columns, _EXPIRY_ALIASES, "expiry")
    forward_col = _require_column(table.columns, _FORWARD_ALIASES, "forward")
    settlement_col = _match_column(table.columns, _SETTLEMENT_ALIASES)
    frame = pd.DataFrame(
        {
            "date": parse_dates(table[date_col], dayfirst=dayfirst),
            "expiry": parse_dates(table[expiry_col], dayfirst=dayfirst),
            "forward": parse_numeric(table[forward_col]),
        }
    )
    if settlement_col is None:
        frame["am_settlement"] = float("nan")
    else:
        frame["am_settlement"] = parse_numeric(table[settlement_col])
    frame = frame.dropna(subset=["date", "expiry", "forward"])
    frame = frame.drop_duplicates(subset=["date", "expiry", "am_settlement"], keep="last")
    return frame.reset_index(drop=True)


def _index_forward_rows(
    frame: pd.DataFrame,
) -> dict[tuple[pd.Timestamp, pd.Timestamp], list[tuple[float, float]]]:
    groups: dict[tuple[pd.Timestamp, pd.Timestamp], list[tuple[float, float]]] = {}
    for date, expiry, settlement, price in frame[
        ["date", "expiry", "am_settlement", "forward"]
    ].itertuples(index=False, name=None):
        key = (pd.Timestamp(date).normalize(), pd.Timestamp(expiry).normalize())
        groups.setdefault(key, []).append((_as_float(settlement), float(price)))
    return groups


def _index_column_forwards(
    rows: pd.DataFrame,
    *,
    dayfirst: bool | None,
) -> dict[pd.Timestamp, list[tuple[float, float]]]:
    expiry_col = _require_column(rows.columns, _EXPIRY_ALIASES, "expiry")
    forward_col = _require_column(rows.columns, (*_FORWARD_ALIASES, "forward_price"), "forward")
    settlement_col = _match_column(rows.columns, _SETTLEMENT_ALIASES)
    frame = pd.DataFrame(
        {
            "expiry": parse_dates(rows[expiry_col], dayfirst=dayfirst),
            "forward": parse_numeric(rows[forward_col]),
        }
    )
    if settlement_col is None:
        frame["am_settlement"] = float("nan")
    else:
        frame["am_settlement"] = parse_numeric(rows[settlement_col])
    frame = frame.dropna(subset=["expiry", "forward"])
    grouped = (
        frame.groupby(["expiry", "am_settlement"], dropna=False, sort=False)["forward"]
        .median()
        .reset_index()
    )
    groups: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    for expiry, settlement, price in grouped.itertuples(index=False, name=None):
        key = pd.Timestamp(expiry).normalize()
        groups.setdefault(key, []).append((_as_float(settlement), float(price)))
    return groups


def _prepare_parity_rows(rows: pd.DataFrame, *, dayfirst: bool | None) -> pd.DataFrame:
    required = ("strike", "cp_flag", "bid", "ask")
    missing = [name for name in required if name not in rows.columns]
    if missing:
        raise SchemaError(
            f"parity quotes missing {missing}; columns {list(rows.columns)}"
        )
    frame = pd.DataFrame(
        {
            "strike": parse_numeric(rows["strike"]),
            "cp_flag": parse_cp_flag(rows["cp_flag"]),
            "bid": parse_numeric(rows["bid"]),
            "ask": parse_numeric(rows["ask"]),
        }
    )
    if "date" in rows.columns:
        frame["date"] = parse_dates(rows["date"], dayfirst=dayfirst)
    if "expiry" in rows.columns:
        frame["expiry"] = parse_dates(rows["expiry"], dayfirst=dayfirst)
    settlement_col = _match_column(rows.columns, _SETTLEMENT_ALIASES)
    if settlement_col is not None:
        frame["am_settlement"] = parse_numeric(rows[settlement_col])
    return frame


def _filter_parity_rows(
    panel: pd.DataFrame,
    date,
    expiry,
    am_settlement: int | None,
    *,
    dayfirst: bool | None,
) -> pd.DataFrame:
    selected = panel
    if "date" in panel.columns:
        selected = selected.loc[selected["date"].eq(_as_timestamp(date, dayfirst=dayfirst))]
    if "expiry" in panel.columns:
        selected = selected.loc[selected["expiry"].eq(_as_timestamp(expiry, dayfirst=dayfirst))]
    if selected.empty:
        return selected
    if am_settlement is not None and "am_settlement" in selected.columns:
        matched = selected.loc[selected["am_settlement"].eq(float(am_settlement))]
        if not matched.empty:
            return matched
    return selected


def _implied_forward(panel: pd.DataFrame, discount: float, spot: float) -> float | None:
    if not np.isfinite(discount) or discount == 0.0:
        return None
    valid = panel.loc[
        panel["bid"].ge(0.0)
        & panel["ask"].gt(0.0)
        & panel["ask"].ge(panel["bid"])
        & panel["strike"].notna()
        & panel["cp_flag"].isin(("C", "P"))
    ].copy()
    if valid.empty:
        return None
    valid["mid"] = (valid["bid"] + valid["ask"]) * 0.5
    calls = valid.loc[valid["cp_flag"].eq("C")].groupby("strike")["mid"].median()
    puts = valid.loc[valid["cp_flag"].eq("P")].groupby("strike")["mid"].median()
    paired = pd.concat({"call": calls, "put": puts}, axis=1).dropna()
    if len(paired) < _PARITY_MIN_STRIKES:
        return None
    strikes = paired.index.to_numpy(dtype=float)
    implied = strikes + (paired["call"].to_numpy(dtype=float) - paired["put"].to_numpy(dtype=float)) / discount
    if np.isfinite(spot):
        reference = float(spot)
    else:
        reference = float(np.median(implied))
    order = np.argsort(np.abs(strikes - reference), kind="mergesort")
    chosen = implied[order[: min(_PARITY_NEAREST, len(order))]]
    return float(np.median(chosen))


def _choose_forward(
    rows: list[tuple[float, float]],
    am_settlement: int | None,
    date: pd.Timestamp | None,
    expiry: pd.Timestamp,
) -> float:
    if len(rows) == 1:
        return rows[0][1]
    has_settlement = any(np.isfinite(settlement) for settlement, _ in rows)
    if has_settlement and am_settlement is not None:
        matched = [
            (settlement, price)
            for settlement, price in rows
            if np.isfinite(settlement) and int(settlement) == am_settlement
        ]
        if len(matched) == 1:
            return matched[0][1]
        if len(matched) > 1:
            logger.debug(
                "multiple forwards for %s %s settlement %s; using PM or first",
                None if date is None else date.date(),
                expiry.date(),
                am_settlement,
            )
            return _pm_or_first(matched)
        logger.debug(
            "no forward for %s %s settlement %s; falling back to date/expiry",
            None if date is None else date.date(),
            expiry.date(),
            am_settlement,
        )
    else:
        logger.debug(
            "multiple forwards for %s %s; using PM or first",
            None if date is None else date.date(),
            expiry.date(),
        )
    return _pm_or_first(rows)


def _pm_or_first(rows: list[tuple[float, float]]) -> float:
    for settlement, price in rows:
        if np.isfinite(settlement) and int(settlement) == 0:
            return price
    return rows[0][1]


def _as_paths(path_or_paths) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        paths = [Path(path_or_paths)]
    else:
        paths = [Path(item) for item in path_or_paths]
    if not paths:
        raise SchemaError("no forward files given")
    return paths


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


def _normalize_settlement(am_settlement: int | None) -> int | None:
    if am_settlement is None:
        return None
    try:
        if bool(pd.isna(am_settlement)):
            return None
    except (TypeError, ValueError):
        pass
    return int(am_settlement)


def _as_float(value) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float("nan")
    try:
        if bool(pd.isna(value)):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float(value)
