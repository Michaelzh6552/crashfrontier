from __future__ import annotations

import csv
import gzip
import importlib
import zipfile
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import NoReturn

import pandas as pd

from crashfrontier.errors import SchemaError

__all__ = [
    "TEXT_SUFFIXES",
    "file_kind",
    "iter_table",
    "peek_columns",
    "read_table",
    "sniff_delimiter",
]

TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}

_COMPRESS_SUFFIXES = {".gz", ".zip"}
_EXCEL_SUFFIXES = {".xls", ".xlsb", ".xlsm", ".xlsx"}
_PARQUET_SUFFIXES = {".parquet"}
_SAS_SUFFIXES = {".sas7bdat"}
_STATA_SUFFIXES = {".dta"}
_SNIFF_BYTES = 64 * 1024
_DEFAULT_CHUNKSIZE = 1_000_000
_TEXT_KINDS = {"text", "text_gz"}
_REPO = "git+https://github.com/Michaelzh6552/crashfrontier.git"
_PARQUET_INSTALL = (
    "reading parquet requires pyarrow: "
    f"pip install \"crashfrontier[parquet] @ {_REPO}\""
)
_EXCEL_OPENPYXL = (
    "reading excel requires openpyxl: "
    f"pip install \"crashfrontier[excel] @ {_REPO}\""
)
_EXCEL_XLS = (
    "reading .xls requires xlrd: "
    f"pip install \"crashfrontier[excel] @ {_REPO}\". "
    "openpyxl cannot read .xls"
)
_EXCEL_XLSB = (
    "reading .xlsb requires pyxlsb: "
    f"pip install \"crashfrontier[excel] @ {_REPO}\". "
    "openpyxl cannot read .xlsb"
)


def file_kind(path: Path) -> str:
    """Classify a path as text, text_gz, parquet, sas, stata, excel, or unknown."""
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    compressed = bool(suffixes) and suffixes[-1] in _COMPRESS_SUFFIXES
    if compressed:
        suffixes = suffixes[:-1]
    if not suffixes:
        return "unknown"
    suffix = suffixes[-1]
    if suffix in TEXT_SUFFIXES:
        return "text_gz" if compressed else "text"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    if suffix in _SAS_SUFFIXES:
        return "sas"
    if suffix in _STATA_SUFFIXES:
        return "stata"
    if suffix in _EXCEL_SUFFIXES:
        return "excel"
    return "unknown"


def sniff_delimiter(path: Path) -> str:
    """Infer the field delimiter from the first 64 KB of a text table."""
    path = Path(path)
    sample = _decode_prefix(_read_prefix(path, _SNIFF_BYTES))
    if "\n" in sample and not sample.endswith("\n"):
        sample = sample.rsplit("\n", 1)[0]
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return ","
    return dialect.delimiter


def peek_columns(path: Path) -> list[str]:
    """Return column names from the file header without loading the body."""
    path = Path(path)
    kind = file_kind(path)
    if kind in _TEXT_KINDS:
        frame = pd.read_csv(path, nrows=0, **_text_csv_kwargs(path, usecols=None))
        return list(frame.columns)
    if kind == "parquet":
        return _peek_parquet(path)
    if kind == "sas":
        return _columns_from_reader(
            pd.read_sas(path, format="sas7bdat", encoding="latin-1", iterator=True)
        )
    if kind == "stata":
        return _columns_from_reader(pd.read_stata(path, iterator=True, chunksize=1))
    if kind == "excel":
        return list(_read_excel(path, usecols=None, nrows=0).columns)
    _unsupported(path)


def read_table(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a table of any supported format into a DataFrame."""
    path = Path(path)
    kind = file_kind(path)
    columns = _resolve_usecols(path, usecols)
    if kind in _TEXT_KINDS:
        return pd.read_csv(path, **_text_csv_kwargs(path, usecols=columns))
    if kind == "parquet":
        return _read_parquet(path, columns=columns)
    if kind == "sas":
        frame = pd.read_sas(path, format="sas7bdat", encoding="latin-1")
        return _take_columns(frame, columns)
    if kind == "stata":
        return pd.read_stata(path, columns=columns)
    if kind == "excel":
        return _read_excel(path, usecols=columns)
    _unsupported(path)


def iter_table(
    path: Path,
    *,
    usecols: list[str] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Iterate over a table in chunks; Excel is yielded once."""
    path = Path(path)
    kind = file_kind(path)
    columns = _resolve_usecols(path, usecols)
    if kind in _TEXT_KINDS:
        yield from _yield_reader(
            pd.read_csv(path, chunksize=chunksize, **_text_csv_kwargs(path, usecols=columns))
        )
        return
    if kind == "parquet":
        yield from _iter_parquet(path, columns=columns)
        return
    if kind == "sas":
        reader = pd.read_sas(
            path, format="sas7bdat", encoding="latin-1", chunksize=chunksize
        )
        for chunk in _yield_reader(reader):
            yield _take_columns(chunk, columns)
        return
    if kind == "stata":
        yield from _yield_reader(
            pd.read_stata(path, columns=columns, chunksize=chunksize)
        )
        return
    if kind == "excel":
        yield _read_excel(path, usecols=columns)
        return
    _unsupported(path)


def _unsupported(path: Path) -> NoReturn:
    raise SchemaError(f"unsupported table file type: {path}")


def _text_csv_kwargs(path: Path, usecols: list[str] | None) -> dict:
    return {
        "sep": sniff_delimiter(path),
        "usecols": usecols,
        "encoding": "utf-8-sig",
        "compression": "infer",
        "low_memory": False,
    }


def _resolve_usecols(path: Path, usecols: list[str] | None) -> list[str] | None:
    if usecols is None:
        return None
    available = peek_columns(path)
    lookup = {name: name for name in available}
    resolved: list[str] = []
    missing: list[str] = []
    for name in usecols:
        matched = lookup.get(name)
        if matched is None:
            missing.append(name)
        else:
            resolved.append(matched)
    if missing:
        raise SchemaError(
            f"columns {missing} are not in {path.name}; available: {available}"
        )
    return resolved


def _take_columns(frame: pd.DataFrame, usecols: list[str] | None) -> pd.DataFrame:
    if usecols is None:
        return frame
    return frame.loc[:, usecols]


def _read_prefix(path: Path, size: int) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read(size)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = _first_zip_member(archive)
            if member is None:
                return b""
            with archive.open(member) as handle:
                return handle.read(size)
    with path.open("rb") as handle:
        return handle.read(size)


def _first_zip_member(archive: zipfile.ZipFile) -> str | None:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    if not names:
        return None
    return names[0]


def _decode_prefix(prefix: bytes) -> str:
    return prefix.decode("utf-8-sig", errors="replace")


def _parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(_PARQUET_INSTALL) from exc
    return pq


def _peek_parquet(path: Path) -> list[str]:
    parquet = _parquet_module()
    return list(parquet.read_schema(path).names)


def _read_parquet(path: Path, columns: list[str] | None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise ImportError(_PARQUET_INSTALL) from exc


def _iter_parquet(path: Path, columns: list[str] | None) -> Iterator[pd.DataFrame]:
    parquet = _parquet_module()
    handle = parquet.ParquetFile(path)
    if handle.num_row_groups == 0:
        yield handle.schema_arrow.empty_table().to_pandas()
        return
    for group_index in range(handle.num_row_groups):
        yield handle.read_row_group(group_index, columns=columns).to_pandas()


def _excel_engine(suffix: str) -> tuple[str, str]:
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl", _EXCEL_OPENPYXL
    if suffix == ".xls":
        return "xlrd", _EXCEL_XLS
    if suffix == ".xlsb":
        return "pyxlsb", _EXCEL_XLSB
    raise SchemaError(f"unsupported excel suffix: {suffix}")


def _import_engine(module: str):
    return importlib.import_module(module)


def _read_excel(
    path: Path, usecols: list[str] | None, nrows: int | None = None
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    engine, message = _excel_engine(suffix)
    try:
        _import_engine(engine)
    except ImportError as exc:
        raise ImportError(message) from exc
    try:
        frame = pd.read_excel(path, usecols=usecols, nrows=nrows, engine=engine)
    except ImportError as exc:
        raise ImportError(message) from exc
    return _excel_preserve_floats(frame)


def _excel_preserve_floats(frame: pd.DataFrame) -> pd.DataFrame:
    # openpyxl stores whole-number floats as integers
    out = frame.copy()
    for name in out.columns:
        if pd.api.types.is_integer_dtype(out[name]):
            out[name] = out[name].astype("float64")
    return out


def _open_reader(reader):
    if hasattr(reader, "__enter__") and hasattr(reader, "__exit__"):
        return reader
    return nullcontext(reader)


def _columns_from_reader(reader) -> list[str]:
    if isinstance(reader, pd.DataFrame):
        return list(reader.columns)
    with _open_reader(reader) as handle:
        names = getattr(handle, "column_names", None)
        if names:
            return [str(name) for name in names]
        labels_fn = getattr(handle, "variable_labels", None)
        if callable(labels_fn):
            labels = labels_fn()
            if labels:
                return list(labels)
        chunk = next(iter(handle), None)
        if chunk is None:
            return []
        return list(chunk.columns)


def _yield_reader(reader) -> Iterator[pd.DataFrame]:
    if isinstance(reader, pd.DataFrame):
        yield reader
        return
    with _open_reader(reader) as handle:
        yield from handle
