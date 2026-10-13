from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from crashfrontier.config import DEFAULT_LM_RANGE, DEFAULT_MEMORY_LIMIT
from crashfrontier.data.cross_section import CrossSection, build_cross_section
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.forwards import (
    ForwardColumn,
    ForwardTable,
    ParityForward,
    resolve_forward,
)
from crashfrontier.data.parse import detect_date_format, detect_strike_scale, parse_dates, parse_numeric
from crashfrontier.data.rates import open_rate_source
from crashfrontier.data.readers import file_kind, peek_columns
from crashfrontier.data.schema import ColumnMap, canonicalize
from crashfrontier.data.source import (
    ChunkScanSource,
    FrameSource,
    IndexedTextSource,
    ParquetSource,
    classify_table,
    open_option_source,
)
from crashfrontier.data.spot import SpotTable
from crashfrontier.errors import MissingForwardError, NoEligibleExpiryError, NoSuchDateError, SchemaError

__all__ = ["Dataset", "classify_table"]

logger = logging.getLogger("crashfrontier")

_DAYS_PER_YEAR = 365.0
_FORWARD_SOURCE_NAMES = ("table", "column", "parity")
_SPOT_EXCLUDE = ("vix", "vx")
_SPOT_PREFER = ("spx", "spot", "secprd", "underlying", "index")
_EXPIRY_COLUMNS = (
    "expiry",
    "am_settlement",
    "root",
    "days",
    "n_rows",
    "n_put_strikes",
    "n_call_strikes",
    "forward",
    "forward_source",
)
_SOURCE_KIND = {
    IndexedTextSource: "indexed_text",
    ParquetSource: "parquet",
    ChunkScanSource: "chunk_scan",
    FrameSource: "frame",
}


class Dataset:
    """Option/forward/rate/spot facade with automatic file classification."""

    def __init__(
        self,
        options,
        *,
        forwards=None,
        rates=None,
        spot=None,
        column_map: ColumnMap | dict[str, str] | None = None,
        strike_scale: float | str = "auto",
        dayfirst: bool | None = None,
        cp_values: dict[str, str] | None = None,
        rate_unit: str = "percent",
        forward_sources: tuple[str, ...] = _FORWARD_SOURCE_NAMES,
        quote_filter: QuoteFilter = QuoteFilter(),
        cache_dir: str | Path | None = None,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> None:
        self._dayfirst = dayfirst
        self._cp_values = None if cp_values is None else dict(cp_values)
        self._rate_unit = rate_unit
        self._forward_names = _validate_forward_sources(forward_sources)
        self._quote_filter = quote_filter
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        self._memory_limit = int(memory_limit)
        self._memory_inputs = _has_frame(options) or _has_frame(forwards) or _has_frame(rates) or _has_frame(spot)
        self._options_ref = None if _has_frame(options) else [str(path) for path in _as_option_paths(options)]
        self._forwards_ref = None if _has_frame(forwards) else _path_ref(forwards)
        self._rates_ref = None if _has_frame(rates) else _rate_ref(rates)
        self._spot_ref = None if _has_frame(spot) else _path_ref(spot)

        mapping = _resolve_column_map(options, column_map)
        self._source, self._column_map = open_option_source(
            options,
            column_map=mapping,
            dayfirst=dayfirst,
            cache_dir=self._cache_dir,
            memory_limit=self._memory_limit,
        )
        self._option_files = _option_file_list(options, self._source)
        self._forward_table = _open_forwards(forwards, dayfirst=dayfirst)
        self._rate_source = open_rate_source(rates, dayfirst=dayfirst, unit=rate_unit)
        self._spot = _open_spot(spot, dayfirst=dayfirst)

        self._auto_strike_scale = isinstance(strike_scale, str)
        if self._auto_strike_scale:
            if strike_scale != "auto":
                raise SchemaError(
                    f"strike_scale must be 'auto' or a number, got {strike_scale!r}"
                )
            self._strike_scale: float | None = None
        else:
            self._strike_scale = float(strike_scale)
        self._date_format: str | None = None
        self._meta_ready = False
        self._cached_stamp: pd.Timestamp | None = None
        self._cached_rows: pd.DataFrame | None = None

    @classmethod
    def from_directory(cls, root: str | Path, **kwargs) -> Dataset:
        """Classify one-level files under ``root`` and optional ``RAW_DATA/``."""
        directory = Path(root)
        if not directory.is_dir():
            raise SchemaError(f"{directory} is not a directory")
        buckets: dict[str, list[Path]] = {}
        for path in _list_directory_tables(directory):
            kind = classify_table(path)
            if kind == "unknown":
                logger.info("ignoring unrecognized table %s", path.name)
                continue
            buckets.setdefault(kind, []).append(path)
        options = buckets.get("options", [])
        if not options:
            raise SchemaError(f"no option tables in directory {directory}")
        forwards = kwargs.pop("forwards", _unique_file(buckets.get("forwards", []), "forwards"))
        rates = kwargs.pop("rates", _rate_file(buckets))
        spot = kwargs.pop("spot", _pick_spot(buckets.get("spot", [])))
        return cls(options, forwards=forwards, rates=rates, spot=spot, **kwargs)

    @classmethod
    def from_frames(
        cls,
        options: pd.DataFrame,
        *,
        forwards=None,
        rates=None,
        spot=None,
        **kwargs,
    ) -> Dataset:
        """Build a dataset from in-memory frames."""
        return cls(options, forwards=forwards, rates=rates, spot=spot, **kwargs)

    @classmethod
    def from_spec(cls, spec: dict) -> Dataset:
        """Rebuild a file-backed dataset from ``spec()``."""
        payload = dict(spec)
        options = payload["options"]
        if isinstance(options, dict):
            options = options.get("paths", options)
        column_map = payload.get("column_map")
        if isinstance(column_map, dict):
            column_map = ColumnMap(**column_map)
        quote_filter = payload.get("quote_filter")
        if isinstance(quote_filter, dict):
            fields = dict(quote_filter)
            if fields.get("cp_flags") is not None:
                fields["cp_flags"] = tuple(fields["cp_flags"])
            quote_filter = QuoteFilter(**fields)
        elif quote_filter is None:
            quote_filter = QuoteFilter()
        forward_sources = payload.get("forward_sources", _FORWARD_SOURCE_NAMES)
        return cls(
            options,
            forwards=payload.get("forwards"),
            rates=payload.get("rates"),
            spot=payload.get("spot"),
            column_map=column_map,
            strike_scale=payload.get("strike_scale", "auto"),
            dayfirst=payload.get("dayfirst"),
            cp_values=payload.get("cp_values"),
            rate_unit=payload.get("rate_unit", "percent"),
            forward_sources=tuple(forward_sources),
            quote_filter=quote_filter,
            cache_dir=payload.get("cache_dir"),
            memory_limit=payload.get("memory_limit", DEFAULT_MEMORY_LIMIT),
        )

    def spec(self) -> dict:
        """JSON-serializable constructor arguments, including resolved strike scale."""
        if self._memory_inputs or self._options_ref is None:
            raise ValueError("in-memory option source cannot be serialized")
        self._ensure_meta()
        payload = {
            "options": list(self._options_ref or []),
            "forwards": self._forwards_ref,
            "rates": self._rates_ref,
            "spot": self._spot_ref,
            "column_map": asdict(self._column_map),
            "strike_scale": float(self._strike_scale),
            "dayfirst": self._dayfirst,
            "cp_values": self._cp_values,
            "rate_unit": self._rate_unit,
            "forward_sources": list(self._forward_names),
            "quote_filter": asdict(self._quote_filter),
            "cache_dir": None if self._cache_dir is None else str(self._cache_dir),
            "memory_limit": int(self._memory_limit),
        }
        return json.loads(json.dumps(payload, default=_json_default))

    def describe(self) -> dict[str, object]:
        """Identified files, date format, strike scale, and auxiliary sources."""
        self._ensure_meta()
        dates = self.available_dates()
        cache = getattr(self._source, "_cache_dir", self._cache_dir)
        return {
            "options": {
                "files": list(self._option_files),
                "kind": _source_kind(self._source),
                "n_dates": len(dates),
                "first_date": None if not dates else dates[0],
                "last_date": None if not dates else dates[-1],
                "columns": self._column_map.rename_map(),
            },
            "date_format": self._date_format,
            "strike_scale": float(self._strike_scale),
            "forwards": None if self._forward_table is None else self._forward_table.name,
            "rates": self._rate_source.name,
            "spot": self._spot is not None,
            "quote_filter": asdict(self._quote_filter),
            "cache_dir": None if cache is None else str(cache),
        }

    def available_dates(self) -> list[pd.Timestamp]:
        """Sorted unique observation dates."""
        return list(self._source.dates())

    def nearest_date(self, date, direction: str = "backward") -> pd.Timestamp:
        """Closest available date at or before/after ``date``."""
        if direction not in {"backward", "forward"}:
            raise ValueError(f"direction must be 'backward' or 'forward', got {direction!r}")
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        dates = self.available_dates()
        if not dates:
            raise NoSuchDateError("option source has no dates")
        if stamp in set(dates):
            return stamp
        if direction == "backward":
            eligible = [item for item in dates if item <= stamp]
            if not eligible:
                raise NoSuchDateError(
                    f"no option date on or before {stamp.date()}; nearest available: {dates[0].date()}"
                )
            return eligible[-1]
        eligible = [item for item in dates if item >= stamp]
        if not eligible:
            raise NoSuchDateError(
                f"no option date on or after {stamp.date()}; nearest available: {dates[-1].date()}"
            )
        return eligible[0]

    def option_rows(self, date) -> pd.DataFrame:
        """Canonical option rows for one observation date."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        if self._cached_stamp == stamp and self._cached_rows is not None:
            return self._cached_rows.copy()
        self._ensure_meta()
        raw = self._source.rows(stamp)
        rows = canonicalize(
            raw,
            self._column_map,
            strike_scale=float(self._strike_scale),
            dayfirst=self._dayfirst,
            cp_values=self._cp_values,
        )
        self._cached_stamp = stamp
        self._cached_rows = rows
        return rows.copy()

    def expiries(self, date, lm_range: tuple[float, float] = DEFAULT_LM_RANGE) -> pd.DataFrame:
        """Per-expiry diagnostics for ``date``; AM and PM stay on separate rows."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        rows = self.option_rows(stamp)
        spot = self._spot_value(stamp)
        records: list[dict[str, object]] = []
        grouped = rows.groupby(["expiry", "am_settlement", "root"], dropna=False, sort=False)
        for (expiry, am_settlement, root), group in grouped:
            expiry_stamp = pd.Timestamp(expiry).normalize()
            settlement = int(am_settlement)
            days = int((expiry_stamp - stamp).days)
            n_rows = int(len(group))
            try:
                forward, source = self._resolve_forward(
                    rows, group, stamp, expiry_stamp, settlement, days, spot
                )
            except MissingForwardError:
                forward, source = float("nan"), "none"
            if np.isfinite(forward):
                filtered, _report = self._quote_filter.apply_rows(group)
                n_put = _n_put_strikes(filtered, forward, lm_range)
                calls = filtered.loc[filtered["cp_flag"].eq("C")]
                n_call = int(calls["strike"].nunique()) if len(calls) else 0
            else:
                n_put = 0
                n_call = 0
            records.append(
                {
                    "expiry": expiry_stamp,
                    "am_settlement": settlement,
                    "root": "" if root is None or (isinstance(root, float) and not np.isfinite(root)) else str(root),
                    "days": days,
                    "n_rows": n_rows,
                    "n_put_strikes": int(n_put),
                    "n_call_strikes": int(n_call),
                    "forward": float(forward),
                    "forward_source": source,
                }
            )
        table = pd.DataFrame.from_records(records, columns=list(_EXPIRY_COLUMNS))
        if table.empty:
            return table
        return table.sort_values(["expiry", "am_settlement", "root"], kind="mergesort").reset_index(drop=True)

    def cross_section(
        self,
        date,
        expiry=None,
        *,
        days=None,
        target_days=None,
        window=None,
        am_settlement=None,
        root=None,
        min_put_strikes: int = 0,
        lm_range: tuple[float, float] = DEFAULT_LM_RANGE,
        forward: float | None = None,
        rate: float | None = None,
    ) -> CrossSection:
        """Select one (date, expiry, settlement) panel and normalize it."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        table = self.expiries(stamp, lm_range=lm_range)
        chosen, ambiguous = self._select_expiry(
            table,
            stamp,
            expiry=expiry,
            days=days,
            target_days=target_days,
            window=window,
            am_settlement=am_settlement,
            root=root,
            min_put_strikes=min_put_strikes,
        )
        rows = self.option_rows(stamp)
        sliced = _slice_group(rows, chosen)
        spot_value = self._spot_value(stamp)
        days_value = int(chosen["days"])
        if forward is not None:
            forward_value, forward_source = float(forward), "user"
        else:
            forward_value, forward_source = float(chosen["forward"]), str(chosen["forward_source"])
        if rate is not None:
            rate_value, rate_source = float(rate), "user"
        else:
            rate_value, rate_source = self._rate_on(stamp, days_value)
        section = build_cross_section(
            sliced,
            date=stamp,
            expiry=chosen["expiry"],
            am_settlement=int(chosen["am_settlement"]),
            forward=forward_value,
            forward_source=forward_source,
            rate=rate_value,
            rate_source=rate_source,
            spot=spot_value,
            quote_filter=self._quote_filter,
        )
        if ambiguous:
            section = replace(section, ambiguous_settlement=True)
        return section

    def cross_sections(self, date, **kwargs) -> list[CrossSection]:
        """Every eligible (expiry, settlement, root) cross-section on ``date``."""
        stamp = _as_timestamp(date, dayfirst=self._dayfirst)
        lm_range = kwargs.get("lm_range", DEFAULT_LM_RANGE)
        table = self.expiries(stamp, lm_range=lm_range)
        candidates = _candidates(
            table,
            min_days=self._quote_filter.min_days,
            max_days=self._quote_filter.max_days,
            am_settlement=kwargs.get("am_settlement"),
            root=kwargs.get("root"),
        )
        sections: list[CrossSection] = []
        for row in candidates.itertuples(index=False):
            sections.append(
                self.cross_section(
                    stamp,
                    expiry=row.expiry,
                    am_settlement=int(row.am_settlement),
                    root=row.root,
                    lm_range=lm_range,
                    forward=kwargs.get("forward"),
                    rate=kwargs.get("rate"),
                )
            )
        return sections

    def _ensure_meta(self) -> None:
        if self._meta_ready:
            return
        dates = self.available_dates()
        if not dates:
            raise SchemaError("option source has no dates")
        first = dates[0]
        raw = self._source.rows(first)
        date_col = self._column_map.date
        if date_col in raw.columns:
            self._date_format = detect_date_format(raw[date_col], dayfirst=self._dayfirst)
        if self._strike_scale is None:
            reference = self._strike_reference(first, raw)
            self._strike_scale = detect_strike_scale(raw[self._column_map.strike], reference)
            logger.info("detected strike_scale=%s", self._strike_scale)
        self._meta_ready = True

    def _strike_reference(self, date: pd.Timestamp, raw: pd.DataFrame) -> float:
        if self._forward_table is not None:
            prices = self._forward_table.prices_on(date)
            if prices:
                median = float(np.median(np.asarray(prices, dtype=float)))
                if np.isfinite(median):
                    return median
        forward_col = self._column_map.forward_price
        if forward_col is not None and forward_col in raw.columns:
            values = parse_numeric(raw[forward_col]).dropna()
            if not values.empty:
                median = float(values.median())
                if np.isfinite(median):
                    return median
        if self._spot is not None:
            value = float(self._spot.spot(date))
            if np.isfinite(value):
                return value
        return float("nan")

    def _spot_value(self, date: pd.Timestamp) -> float:
        if self._spot is None:
            return float("nan")
        return float(self._spot.spot(date))

    def _rate_on(self, date: pd.Timestamp, days: int) -> tuple[float, str]:
        return float(self._rate_source.rate(date, int(days))), self._rate_source.name

    def _resolve_forward(
        self,
        day_rows: pd.DataFrame,
        group_rows: pd.DataFrame,
        date: pd.Timestamp,
        expiry: pd.Timestamp,
        am_settlement: int,
        days: int,
        spot: float,
    ) -> tuple[float, str]:
        chain = []
        for name in self._forward_names:
            if name == "table":
                if self._forward_table is not None:
                    chain.append(self._forward_table)
            elif name == "column":
                chain.append(ForwardColumn(day_rows, dayfirst=self._dayfirst))
            elif name == "parity":
                try:
                    rate_value, _source = self._rate_on(date, days)
                except ValueError:
                    continue
                discount = math.exp(-rate_value * days / _DAYS_PER_YEAR)
                chain.append(ParityForward(group_rows, discount, spot, dayfirst=self._dayfirst))
        if not chain:
            raise MissingForwardError(
                f"no forward sources configured for {date.date()} {expiry.date()}"
            )
        return resolve_forward(chain, date, expiry, am_settlement)

    def _select_expiry(
        self,
        table: pd.DataFrame,
        date: pd.Timestamp,
        *,
        expiry,
        days,
        target_days,
        window,
        am_settlement,
        root,
        min_put_strikes: int,
    ) -> tuple[pd.Series, bool]:
        candidates = _candidates(
            table,
            min_days=self._quote_filter.min_days,
            max_days=self._quote_filter.max_days,
            am_settlement=am_settlement,
            root=root,
        )
        if expiry is not None:
            stamp = _as_timestamp(expiry, dayfirst=self._dayfirst)
            matched = candidates.loc[candidates["expiry"].eq(stamp)]
        elif days is not None:
            matched = candidates.loc[candidates["days"].eq(int(days))]
        elif target_days is not None:
            lo, hi = (
                (self._quote_filter.min_days, self._quote_filter.max_days)
                if window is None
                else (int(window[0]), int(window[1]))
            )
            matched = candidates.loc[
                candidates["days"].between(lo, hi)
                & candidates["n_put_strikes"].ge(int(min_put_strikes))
            ]
            if not matched.empty:
                delta = (matched["days"] - int(target_days)).abs()
                matched = matched.loc[delta.eq(delta.min())]
                earliest = matched["expiry"].min()
                matched = matched.loc[matched["expiry"].eq(earliest)]
        else:
            raise ValueError("cross_section requires expiry, days, or target_days")
        if matched.empty:
            raise NoEligibleExpiryError(
                f"no eligible expiry for {date.date()}; available:\n{table.to_string(index=False)}"
            )
        ambiguous = False
        if len(matched) > 1 and am_settlement is None and matched["am_settlement"].nunique() > 1:
            ambiguous = True
            ordered = matched.sort_values(
                ["n_put_strikes", "am_settlement"],
                ascending=[False, True],
                kind="mergesort",
            )
            chosen = ordered.iloc[0]
            logger.warning(
                "ambiguous AM/PM settlement for %s expiry %s; "
                "selected am_settlement=%s with %s put strikes",
                date.date(),
                pd.Timestamp(chosen["expiry"]).date(),
                int(chosen["am_settlement"]),
                int(chosen["n_put_strikes"]),
            )
            return chosen, True
        if len(matched) > 1:
            matched = matched.sort_values(
                ["n_put_strikes", "am_settlement", "root"],
                ascending=[False, True, True],
                kind="mergesort",
            )
        return matched.iloc[0], ambiguous


def _list_directory_tables(root: Path) -> list[Path]:
    directories = [root]
    nested = root / "RAW_DATA"
    if nested.is_dir():
        directories.append(nested)
    files: list[Path] = []
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                files.append(path)
    return files


def _unique_file(paths: list[Path], label: str) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1:
        names = ", ".join(path.name for path in paths)
        raise SchemaError(f"multiple {label} tables ({names}); pass {label}= explicitly")
    return paths[0]


def _rate_file(buckets: dict[str, list[Path]]) -> Path | None:
    zero = buckets.get("zero_curve", [])
    series = buckets.get("rate_series", [])
    if zero and series:
        raise SchemaError("multiple rate tables; pass rates= explicitly")
    if zero:
        return _unique_file(zero, "rates")
    return _unique_file(series, "rates")


def _pick_spot(paths: list[Path]) -> Path | None:
    remaining = [path for path in paths if not _name_contains(path, _SPOT_EXCLUDE)]
    if len(remaining) > 1:
        preferred = [path for path in remaining if _name_contains(path, _SPOT_PREFER)]
        if preferred:
            remaining = preferred
    if not remaining:
        return None
    if len(remaining) > 1:
        names = ", ".join(path.name for path in remaining)
        raise SchemaError(f"multiple spot tables ({names}); pass spot= explicitly")
    return remaining[0]


def _name_contains(path: Path, tokens: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(token in name for token in tokens)


def _resolve_column_map(options, column_map) -> ColumnMap | None:
    if isinstance(column_map, ColumnMap) or column_map is None:
        return column_map
    if not isinstance(column_map, dict):
        raise SchemaError(f"column_map must be ColumnMap or dict, got {type(column_map)!r}")
    return ColumnMap.detect(_peek_option_columns(options), overrides=column_map)


def _peek_option_columns(options) -> list[str]:
    if isinstance(options, pd.DataFrame):
        return [str(column) for column in options.columns]
    return peek_columns(_as_option_paths(options)[0])


def _as_option_paths(options) -> list[Path]:
    if isinstance(options, (str, Path)):
        path = Path(options)
        if path.is_dir():
            files = [
                item
                for item in sorted(path.iterdir())
                if item.is_file() and file_kind(item) != "unknown" and classify_table(item) == "options"
            ]
            if not files:
                raise SchemaError(f"no option tables in directory {path}")
            return files
        return [path]
    paths = [Path(item) for item in options]
    if not paths:
        raise SchemaError("no option files given")
    return paths


def _option_file_list(options, source) -> list[str]:
    paths = getattr(source, "_paths", None)
    if paths:
        return [str(path) for path in paths]
    if isinstance(options, pd.DataFrame):
        return []
    return [str(path) for path in _as_option_paths(options)]


def _open_forwards(forwards, *, dayfirst: bool | None) -> ForwardTable | None:
    if forwards is None:
        return None
    if isinstance(forwards, ForwardTable):
        return forwards
    if isinstance(forwards, pd.DataFrame):
        return ForwardTable.from_frame(forwards, dayfirst=dayfirst)
    return ForwardTable.from_path(forwards, dayfirst=dayfirst)


def _open_spot(spot, *, dayfirst: bool | None) -> SpotTable | None:
    if spot is None:
        return None
    if isinstance(spot, SpotTable):
        return spot
    if isinstance(spot, pd.DataFrame):
        return SpotTable(spot, dayfirst=dayfirst)
    return SpotTable.from_path(spot, dayfirst=dayfirst)


def _validate_forward_sources(names: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(str(name) for name in names)
    unknown = [name for name in resolved if name not in _FORWARD_SOURCE_NAMES]
    if unknown:
        raise SchemaError(
            f"unknown forward source {unknown[0]!r}; expected {list(_FORWARD_SOURCE_NAMES)}"
        )
    return resolved


def _has_frame(value) -> bool:
    return isinstance(value, pd.DataFrame)


def _path_ref(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    return [str(Path(item)) for item in value]


def _rate_ref(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise SchemaError("boolean is not a rate source")
    if isinstance(value, (int, float)):
        return float(value)
    return _path_ref(value)


def _source_kind(source) -> str:
    for cls, name in _SOURCE_KIND.items():
        if isinstance(source, cls):
            return name
    return "frame"


def _as_timestamp(value, *, dayfirst: bool | None = None) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize(None) if value.tzinfo is not None else value
        return stamp.normalize()
    parsed = parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0]
    if pd.isna(parsed):
        raise SchemaError(f"could not parse date {value!r}")
    return pd.Timestamp(parsed).normalize()


def _candidates(
    table: pd.DataFrame,
    *,
    min_days: int,
    max_days: int,
    am_settlement,
    root,
) -> pd.DataFrame:
    if table.empty:
        return table
    selected = table.loc[np.isfinite(table["forward"].to_numpy(dtype=float))]
    selected = selected.loc[selected["days"].between(int(min_days), int(max_days))]
    if am_settlement is not None:
        selected = selected.loc[selected["am_settlement"].eq(int(am_settlement))]
    if root is not None:
        selected = selected.loc[selected["root"].eq(str(root))]
    return selected


def _slice_group(rows: pd.DataFrame, chosen: pd.Series) -> pd.DataFrame:
    expiry = pd.Timestamp(chosen["expiry"]).normalize()
    settlement = int(chosen["am_settlement"])
    root = str(chosen["root"])
    sliced = rows.loc[
        rows["expiry"].eq(expiry)
        & rows["am_settlement"].eq(settlement)
        & rows["root"].eq(root)
    ]
    return sliced.reset_index(drop=True)


def _n_put_strikes(rows: pd.DataFrame, forward: float, lm_range: tuple[float, float]) -> int:
    if rows.empty or not np.isfinite(forward) or forward <= 0.0:
        return 0
    puts = rows.loc[rows["cp_flag"].eq("P")]
    if puts.empty:
        return 0
    strikes = puts["strike"].to_numpy(dtype=float, copy=False)
    log_m = np.full(strikes.shape, np.nan)
    positive = strikes > 0.0
    log_m[positive] = np.log(strikes[positive] / float(forward))
    lo, hi = float(lm_range[0]), float(lm_range[1])
    keep = np.isfinite(log_m) & (log_m >= lo) & (log_m <= hi)
    return int(pd.unique(strikes[keep]).size)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")
