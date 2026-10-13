from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

__all__ = [
    "DateIndex",
    "build_date_index",
    "load_date_index",
    "read_rows_by_ranges",
]

_PROGRESS_EVERY = 1_000_000
_MTIME_ABS_TOL = 1e-6


@dataclass(frozen=True)
class DateIndex:
    source: Path
    size: int
    mtime: float
    delimiter: str
    header: bytes
    date_column: int
    ranges: dict[str, list[tuple[int, int]]]


def build_date_index(
    csv_path: Path, index_path: Path, *, date_column: int, delimiter: str
) -> DateIndex:
    """Scan an uncompressed text table once and record byte ranges per date."""
    csv_path = Path(csv_path).resolve()
    index_path = Path(index_path)
    size, mtime = _stat(csv_path)
    delim = delimiter.encode()
    ranges: dict[str, list[tuple[int, int]]] = {}
    current_key = ""
    start = 0
    n_lines = 0
    last_update = 0

    with csv_path.open("rb") as handle:
        header = handle.readline()
        position = len(header)
        start = position
        with tqdm(total=size, unit="B", unit_scale=True, disable=False) as bar:
            if position:
                bar.update(position)
                last_update = position
            for line in handle:
                parts = line.split(delim, date_column + 1)
                if len(parts) <= date_column:
                    key = ""
                else:
                    key = parts[date_column].strip().strip(b'"').decode("utf-8", "replace")
                if key != current_key:
                    if current_key:
                        ranges.setdefault(current_key, []).append((start, position))
                    current_key = key
                    start = position
                position += len(line)
                n_lines += 1
                if n_lines % _PROGRESS_EVERY == 0:
                    bar.update(position - last_update)
                    last_update = position
            if current_key:
                ranges.setdefault(current_key, []).append((start, position))
            remaining = bar.total - bar.n
            if remaining > 0:
                bar.update(remaining)

    index = DateIndex(
        source=csv_path,
        size=size,
        mtime=mtime,
        delimiter=delimiter,
        header=header,
        date_column=date_column,
        ranges=ranges,
    )
    _write_index(index_path, index)
    return index


def load_date_index(
    csv_path: Path, index_path: Path, *, date_column: int, delimiter: str
) -> DateIndex:
    """Load a date index, rebuilding when source, size, mtime, column, or delimiter disagree."""
    csv_path = Path(csv_path)
    index_path = Path(index_path)
    size, mtime = _stat(csv_path)
    if index_path.is_file():
        try:
            index = _read_index(index_path)
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            index = None
        else:
            if (
                _same_resolved_path(index.source, csv_path)
                and index.size == size
                and _mtime_matches(index.mtime, mtime)
                and index.date_column == date_column
                and index.delimiter == delimiter
            ):
                return index
    return build_date_index(
        csv_path, index_path, date_column=date_column, delimiter=delimiter
    )


def read_rows_by_ranges(
    index: DateIndex, raw_date: str, usecols: list[str] | None
) -> pd.DataFrame:
    """Read the raw CSV rows whose byte ranges belong to ``raw_date``."""
    spans = index.ranges.get(raw_date, [])
    chunks = [index.header]
    if spans:
        with Path(index.source).open("rb") as handle:
            for start, end in spans:
                handle.seek(start)
                chunks.append(handle.read(end - start))
    payload = b"".join(chunks)
    return pd.read_csv(
        io.BytesIO(payload),
        sep=index.delimiter,
        usecols=usecols,
        low_memory=False,
        encoding="utf-8-sig",
    )


def _stat(path: Path) -> tuple[int, float]:
    info = Path(path).stat()
    return int(info.st_size), float(info.st_mtime)


def _mtime_matches(stored: float, actual: float) -> bool:
    return math.isclose(stored, actual, rel_tol=0.0, abs_tol=_MTIME_ABS_TOL)


def _same_resolved_path(stored: Path, csv_path: Path) -> bool:
    try:
        return Path(stored).resolve() == Path(csv_path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _write_index(index_path: Path, index: DateIndex) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(index.source),
        "size": index.size,
        "mtime": index.mtime,
        "delimiter": index.delimiter,
        "header": index.header.decode("latin-1"),
        "date_column": index.date_column,
        "ranges": {
            key: [[start, end] for start, end in spans] for key, spans in index.ranges.items()
        },
    }
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(index_path)


def _read_index(index_path: Path) -> DateIndex:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    ranges: dict[str, list[tuple[int, int]]] = {
        str(key): [(int(start), int(end)) for start, end in spans]
        for key, spans in payload["ranges"].items()
    }
    return DateIndex(
        source=Path(payload["source"]).resolve(),
        size=int(payload["size"]),
        mtime=float(payload["mtime"]),
        delimiter=str(payload["delimiter"]),
        header=str(payload["header"]).encode("latin-1"),
        date_column=int(payload["date_column"]),
        ranges=ranges,
    )
