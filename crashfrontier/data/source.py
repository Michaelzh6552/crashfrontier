from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from crashfrontier.config import DEFAULT_MEMORY_LIMIT
from crashfrontier.data.index import load_date_index, read_rows_by_ranges
from crashfrontier.data.parse import parse_dates
from crashfrontier.data.readers import file_kind, iter_table, peek_columns, read_table, sniff_delimiter
from crashfrontier.data.schema import ALIASES, ColumnMap, normalize_name
from crashfrontier.errors import NoSuchDateError, SchemaError

__all__ = [
    "ChunkScanSource",
    "FrameSource",
    "IndexedTextSource",
    "OptionSource",
    "ParquetSource",
    "classify_table",
    "open_option_source",
]

logger = logging.getLogger("crashfrontier")

_CHUNK_KINDS = frozenset({"text_gz", "sas", "stata", "excel"})
_FORWARD_ALIASES = ("forwardprice", "forward_price", "forward", "fwd", "fwd_price", "f")
_ZERO_RATE_ALIASES = ("rate", "zero_rate", "yield", "zero")
_SPOT_PRICE_ALIASES = ("close", "price", "adj_close", "adjclose", "spindx", "level", "last", "value")
_RATE_SERIES_VALUE_ALIASES = ("rate", "yield", "tbill", "riskfree", "rf", "value")
_MTIME_ABS_TOL = 1e-6
_DATE_HELPER = "_date"


@runtime_checkable
class OptionSource(Protocol):
    columns: list[str]

    def dates(self) -> list[pd.Timestamp]: ...

    def rows(self, date: pd.Timestamp) -> pd.DataFrame: ...

    def spec(self) -> dict: ...


class FrameSource:
    """In-memory option table."""

    def __init__(
        self,
        frame: pd.DataFrame,
        column_map: ColumnMap,
        *,
        dayfirst: bool | None = None,
    ) -> None:
        self.columns = [str(column) for column in frame.columns]
        self._column_map = column_map
        self._dayfirst = dayfirst
        self._frame = frame.copy()
        self._frame[_DATE_HELPER] = parse_dates(self._frame[column_map.date], dayfirst=dayfirst)
        unique = self._frame[_DATE_HELPER].dropna().unique()
        self._dates = sorted(pd.Timestamp(stamp).normalize() for stamp in unique)

    def dates(self) -> list[pd.Timestamp]:
        return list(self._dates)

    def rows(self, date: pd.Timestamp) -> pd.DataFrame:
        stamp = _coerce_timestamp(date, dayfirst=self._dayfirst)
        matched = self._frame.loc[self._frame[_DATE_HELPER].eq(stamp), self.columns]
        if matched.empty:
            raise _no_such_date(stamp, self._dates)
        return matched.copy()

    def spec(self) -> dict:
        raise ValueError("in-memory option source cannot be serialized")


class ParquetSource:
    """Parquet option tables, filtered by the original date column value."""

    def __init__(
        self,
        paths: Sequence[Path],
        column_map: ColumnMap,
        *,
        dayfirst: bool | None = None,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> None:
        self._paths = [Path(path) for path in paths]
        self._column_map = column_map
        self._dayfirst = dayfirst
        self._memory_limit = memory_limit
        self.columns = peek_columns(self._paths[0])
        self._raw_by_file: dict[Path, dict[pd.Timestamp, list[object]]] = {}
        self._date_col_by_file: dict[Path, str] = {}
        dates: set[pd.Timestamp] = set()
        for path in self._paths:
            date_col = _resolve_column(peek_columns(path), column_map.date)
            raw = pd.read_parquet(path, columns=[date_col])[date_col]
            parsed = parse_dates(raw, dayfirst=dayfirst)
            mapping: dict[pd.Timestamp, list[object]] = {}
            for stamp, value in zip(parsed, raw, strict=False):
                if pd.isna(stamp):
                    continue
                key = pd.Timestamp(stamp).normalize()
                token = _parquet_filter_value(value)
                bucket = mapping.setdefault(key, [])
                if token not in bucket:
                    bucket.append(token)
                dates.add(key)
            self._raw_by_file[path] = mapping
            self._date_col_by_file[path] = date_col
        self._dates = sorted(dates)

    def dates(self) -> list[pd.Timestamp]:
        return list(self._dates)

    def rows(self, date: pd.Timestamp) -> pd.DataFrame:
        stamp = _coerce_timestamp(date, dayfirst=self._dayfirst)
        frames: list[pd.DataFrame] = []
        for path in self._paths:
            raw_values = self._raw_by_file[path].get(stamp)
            if not raw_values:
                continue
            date_col = self._date_col_by_file[path]
            piece = pd.read_parquet(
                path, filters=[(date_col, "in", raw_values)]
            )
            frames.append(_align_frame(piece, self.columns))
        if not frames:
            raise _no_such_date(stamp, self._dates)
        return pd.concat(frames, ignore_index=True)

    def spec(self) -> dict:
        return {
            "kind": "parquet",
            "paths": [str(path) for path in self._paths],
            "column_map": asdict(self._column_map),
            "dayfirst": self._dayfirst,
            "memory_limit": self._memory_limit,
        }


class ChunkScanSource:
    """Scan compressed or binary tables in chunks; cache unique dates."""

    def __init__(
        self,
        paths: Sequence[Path],
        column_map: ColumnMap,
        *,
        cache_dir: Path,
        dayfirst: bool | None = None,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> None:
        self._paths = [Path(path) for path in paths]
        self._column_map = column_map
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dayfirst = dayfirst
        self._memory_limit = memory_limit
        self.columns = peek_columns(self._paths[0])
        self._dates_by_file: dict[Path, set[pd.Timestamp]] = {}
        logger.warning(
            "chunked scans reread the entire compressed file; "
            "convert it to uncompressed CSV for indexed access"
        )
        dates: set[pd.Timestamp] = set()
        cache_names = _cache_filenames(self._paths, "dates.json")
        for path in self._paths:
            found = _load_or_scan_dates(
                path,
                column_map=column_map,
                cache_path=self._cache_dir / cache_names[path],
                dayfirst=dayfirst,
            )
            self._dates_by_file[path] = found
            dates.update(found)
        self._dates = sorted(dates)

    def dates(self) -> list[pd.Timestamp]:
        return list(self._dates)

    def rows(self, date: pd.Timestamp) -> pd.DataFrame:
        stamp = _coerce_timestamp(date, dayfirst=self._dayfirst)
        frames: list[pd.DataFrame] = []
        for path in self._paths:
            if stamp not in self._dates_by_file[path]:
                continue
            date_col = _resolve_column(peek_columns(path), self._column_map.date)
            for chunk in iter_table(path):
                parsed = parse_dates(chunk[date_col], dayfirst=self._dayfirst)
                hit = chunk.loc[parsed.eq(stamp)]
                if not hit.empty:
                    frames.append(_align_frame(hit, self.columns))
        if not frames:
            raise _no_such_date(stamp, self._dates)
        return pd.concat(frames, ignore_index=True)

    def spec(self) -> dict:
        return {
            "kind": "chunk_scan",
            "paths": [str(path) for path in self._paths],
            "cache_dir": str(self._cache_dir),
            "column_map": asdict(self._column_map),
            "dayfirst": self._dayfirst,
            "memory_limit": self._memory_limit,
        }


class IndexedTextSource:
    """Uncompressed text tables addressed by a byte-range date index."""

    def __init__(
        self,
        paths: Sequence[Path],
        column_map: ColumnMap,
        *,
        cache_dir: Path,
        dayfirst: bool | None = None,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> None:
        self._paths = [Path(path) for path in paths]
        self._column_map = column_map
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dayfirst = dayfirst
        self._memory_limit = memory_limit
        self.columns = peek_columns(self._paths[0])
        self._indexes = {}
        owners: list[tuple[Path, str]] = []
        keys: list[str] = []
        cache_names = _cache_filenames(self._paths, "index.json")
        for path in self._paths:
            columns = peek_columns(path)
            date_name = _resolve_column(columns, column_map.date)
            date_column = columns.index(date_name)
            delimiter = sniff_delimiter(path)
            index_path = self._cache_dir / cache_names[path]
            index = load_date_index(
                path, index_path, date_column=date_column, delimiter=delimiter
            )
            self._indexes[path] = index
            for key in index.ranges:
                owners.append((path, key))
                keys.append(key)
        mapping: dict[pd.Timestamp, list[tuple[Path, str]]] = {}
        if keys:
            parsed = parse_dates(pd.Series(keys), dayfirst=dayfirst)
            for stamp, owner in zip(parsed, owners, strict=False):
                if pd.isna(stamp):
                    continue
                mapping.setdefault(pd.Timestamp(stamp).normalize(), []).append(owner)
        self._map = mapping
        self._dates = sorted(mapping)

    def dates(self) -> list[pd.Timestamp]:
        return list(self._dates)

    def rows(self, date: pd.Timestamp) -> pd.DataFrame:
        stamp = _coerce_timestamp(date, dayfirst=self._dayfirst)
        owners = self._map.get(stamp)
        if not owners:
            raise _no_such_date(stamp, self._dates)
        frames = [
            _align_frame(read_rows_by_ranges(self._indexes[path], key, None), self.columns)
            for path, key in owners
        ]
        return pd.concat(frames, ignore_index=True)

    def spec(self) -> dict:
        return {
            "kind": "indexed_text",
            "paths": [str(path) for path in self._paths],
            "cache_dir": str(self._cache_dir),
            "column_map": asdict(self._column_map),
            "dayfirst": self._dayfirst,
            "memory_limit": self._memory_limit,
        }


def open_option_source(
    options,
    *,
    column_map: ColumnMap | None = None,
    dayfirst: bool | None = None,
    cache_dir: str | Path | None = None,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
) -> tuple[OptionSource, ColumnMap]:
    """Open option quotes from a frame, path, path list, or directory."""
    if isinstance(options, pd.DataFrame):
        mapping = column_map or ColumnMap.detect(list(options.columns))
        return FrameSource(options, mapping, dayfirst=dayfirst), mapping

    paths = _resolve_option_paths(options)
    _assert_compatible_columns(paths)
    mapping = column_map or ColumnMap.detect(peek_columns(paths[0]))
    kinds = [file_kind(path) for path in paths]
    total_size = sum(path.stat().st_size for path in paths)
    resolved_cache = _resolve_cache_dir(cache_dir, paths)

    if all(kind == "text" for kind in kinds) and total_size > memory_limit:
        source: OptionSource = IndexedTextSource(
            paths,
            mapping,
            cache_dir=resolved_cache,
            dayfirst=dayfirst,
            memory_limit=memory_limit,
        )
        return source, mapping
    if total_size <= memory_limit:
        reference = peek_columns(paths[0])
        frames = [_align_frame(read_table(path), reference) for path in paths]
        source = FrameSource(pd.concat(frames, ignore_index=True), mapping, dayfirst=dayfirst)
        return source, mapping
    if all(kind == "parquet" for kind in kinds):
        source = ParquetSource(
            paths, mapping, dayfirst=dayfirst, memory_limit=memory_limit
        )
        return source, mapping
    if all(kind in _CHUNK_KINDS for kind in kinds):
        source = ChunkScanSource(
            paths,
            mapping,
            cache_dir=resolved_cache,
            dayfirst=dayfirst,
            memory_limit=memory_limit,
        )
        return source, mapping
    raise SchemaError(f"mixed large option file types: {kinds}")


def classify_table(path: Path) -> str:
    """Classify a table by header aliases; does not read the body."""
    path = Path(path)
    if file_kind(path) == "unknown":
        return "unknown"
    columns = peek_columns(path)
    normalized = {normalize_name(column) for column in columns}
    if (
        _alias_hit(normalized, ALIASES["strike"])
        and _alias_hit(normalized, ALIASES["cp_flag"])
        and _alias_hit(normalized, ALIASES["bid"])
    ):
        return "options"
    if (
        _alias_hit(normalized, _FORWARD_ALIASES)
        and _alias_hit(normalized, ALIASES["expiry"])
        and not _alias_hit(normalized, ALIASES["strike"])
    ):
        return "forwards"
    if "days" in normalized and _alias_hit(normalized, _ZERO_RATE_ALIASES):
        return "zero_curve"
    if len(columns) == 2:
        date_hits = [_alias_hit({normalize_name(column)}, ALIASES["date"]) for column in columns]
        if sum(date_hits) == 1:
            other = columns[1] if date_hits[0] else columns[0]
            other_name = normalize_name(other)
            value_names = {normalize_name(alias) for alias in _RATE_SERIES_VALUE_ALIASES}
            if other_name.startswith("dgs") or other_name in value_names:
                return "rate_series"
    if _alias_hit(normalized, ALIASES["date"]) and _alias_hit(normalized, _SPOT_PRICE_ALIASES):
        return "spot"
    return "unknown"


def _resolve_option_paths(options) -> list[Path]:
    if isinstance(options, (str, Path)):
        path = Path(options)
        if path.is_dir():
            files = _directory_option_files(path)
            if not files:
                raise SchemaError(f"no option tables in directory {path}")
            return files
        return [path]
    paths = [Path(path) for path in options]
    if not paths:
        raise SchemaError("no option files given")
    return paths


def _directory_option_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if file_kind(path) == "unknown":
            continue
        if classify_table(path) == "options":
            files.append(path)
    return files


def _assert_compatible_columns(paths: Sequence[Path]) -> None:
    expected = {normalize_name(column) for column in peek_columns(paths[0])}
    for path in paths[1:]:
        actual = {normalize_name(column) for column in peek_columns(path)}
        if actual != expected:
            raise SchemaError(
                f"option file columns do not match: {paths[0].name} vs {path.name}"
            )


def _resolve_cache_dir(cache_dir: str | Path | None, paths: Sequence[Path]) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return paths[0].resolve().parent.parent / "CACHE"


def _resolve_column(columns: Sequence[str], original: str) -> str:
    names = [str(column) for column in columns]
    if original in names:
        return original
    key = normalize_name(original)
    for name in names:
        if normalize_name(name) == key:
            return name
    raise SchemaError(f"column {original!r} is not among {names}")


def _align_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    lookup = {normalize_name(column): column for column in frame.columns}
    rename: dict[str, str] = {}
    ordered: list[str] = []
    missing: list[str] = []
    for column in columns:
        source = lookup.get(normalize_name(column))
        if source is None:
            missing.append(str(column))
            continue
        if source != column:
            rename[source] = str(column)
        ordered.append(str(column))
    if missing:
        raise SchemaError(f"columns {missing} are missing from aligned frame")
    aligned = frame.rename(columns=rename) if rename else frame
    extra = [column for column in aligned.columns if str(column) not in ordered]
    return aligned.loc[:, list(ordered) + extra]


def _alias_hit(normalized: set[str], aliases: Sequence[str]) -> bool:
    return any(normalize_name(alias) in normalized for alias in aliases)


def _coerce_timestamp(value, *, dayfirst: bool | None) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize(None) if value.tzinfo is not None else value
        return stamp.normalize()
    parsed = parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0]
    if pd.isna(parsed):
        raise NoSuchDateError(f"could not parse option date {value!r}")
    return pd.Timestamp(parsed).normalize()


def _no_such_date(requested: pd.Timestamp, available: Sequence[pd.Timestamp]) -> NoSuchDateError:
    if not available:
        return NoSuchDateError(
            f"no option rows for {requested.date()}; source has no dates"
        )
    before = None
    after = None
    for stamp in available:
        if stamp < requested:
            before = stamp
        elif stamp > requested:
            after = stamp
            break
    neighbors = []
    if before is not None:
        neighbors.append(str(before.date()))
    if after is not None:
        neighbors.append(str(after.date()))
    nearest = ", ".join(neighbors) if neighbors else "none"
    return NoSuchDateError(
        f"no option rows for {requested.date()}; nearest available: {nearest}"
    )


def _parquet_filter_value(value):
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if hasattr(value, "dtype") and "datetime64" in str(getattr(value, "dtype", "")):
        return pd.Timestamp(value).to_pydatetime()
    return value


def _stat(path: Path) -> tuple[int, float]:
    info = Path(path).stat()
    return int(info.st_size), float(info.st_mtime)


def _mtime_matches(stored: float, actual: float) -> bool:
    return math.isclose(stored, actual, rel_tol=0.0, abs_tol=_MTIME_ABS_TOL)


def _load_or_scan_dates(
    path: Path,
    *,
    column_map: ColumnMap,
    cache_path: Path,
    dayfirst: bool | None,
) -> set[pd.Timestamp]:
    source = str(path.resolve())
    size, mtime = _stat(path)
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if _dates_cache_current(
                payload, source=source, size=size, mtime=mtime, dayfirst=dayfirst
            ):
                return {pd.Timestamp(item).normalize() for item in payload["dates"]}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    date_col = _resolve_column(peek_columns(path), column_map.date)
    found: set[pd.Timestamp] = set()
    for chunk in iter_table(path, usecols=[date_col]):
        parsed = parse_dates(chunk[date_col], dayfirst=dayfirst)
        for stamp in parsed.dropna():
            found.add(pd.Timestamp(stamp).normalize())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "size": size,
        "mtime": mtime,
        "dayfirst": dayfirst,
        "dates": [stamp.strftime("%Y-%m-%d") for stamp in sorted(found)],
    }
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(cache_path)
    return found


def _dates_cache_current(
    payload: object,
    *,
    source: str,
    size: int,
    mtime: float,
    dayfirst: bool | None,
) -> bool:
    if not isinstance(payload, dict) or "dayfirst" not in payload or "source" not in payload:
        return False
    return (
        _same_resolved_path(payload["source"], source)
        and int(payload["size"]) == size
        and _mtime_matches(float(payload["mtime"]), mtime)
        and payload["dayfirst"] == dayfirst
    )


def _cache_filenames(paths: Sequence[Path], suffix: str) -> dict[Path, str]:
    """Unique stems keep ``<stem>.<suffix>``; duplicates add a resolved-path hash."""
    tally: dict[str, int] = {}
    for path in paths:
        tally[path.stem] = tally.get(path.stem, 0) + 1
    return {
        path: _cache_filename(path, suffix, shared_stem=tally[path.stem] > 1)
        for path in paths
    }


def _cache_filename(path: Path, suffix: str, *, shared_stem: bool) -> str:
    stem = path.stem
    if not shared_stem:
        return f"{stem}.{suffix}"
    return f"{stem}-{_resolved_path_token(path)}.{suffix}"


def _resolved_path_token(path: Path) -> str:
    canonical = str(Path(path).resolve()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _same_resolved_path(stored: object, path: Path | str) -> bool:
    try:
        return Path(str(stored)).resolve() == Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
