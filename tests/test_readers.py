from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from crashfrontier.data.readers import (
    TEXT_SUFFIXES,
    file_kind,
    iter_table,
    peek_columns,
    read_table,
    sniff_delimiter,
)
from crashfrontier.errors import SchemaError


def _quotes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-22", "2021-07-28", "2021-07-29", "2021-07-30"],
            "cp_flag": ["P", "C", "P", "C", "P"],
            "strike": [4000.0, 4100.0, 4200.0, 4300.0, 4400.0],
            "bid": [1.25, 2.5, 3.75, 4.0, 5.25],
            "ask": [1.35, 2.6, 3.85, 4.1, 5.35],
        }
    )


def _write_delimited(path: Path, frame: pd.DataFrame, sep: str) -> None:
    frame.to_csv(path, index=False, sep=sep)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("quotes.csv", "text"),
        ("quotes.CSV", "text"),
        ("quotes.txt", "text"),
        ("quotes.tsv", "text"),
        ("quotes.dat", "text"),
        ("quotes.csv.gz", "text_gz"),
        ("quotes.CSV.GZ", "text_gz"),
        ("quotes.txt.zip", "text_gz"),
        ("quotes.csv.zip", "text_gz"),
        ("quotes.parquet", "parquet"),
        ("quotes.parquet.gz", "parquet"),
        ("quotes.sas7bdat", "sas"),
        ("quotes.dta", "stata"),
        ("quotes.xlsx", "excel"),
        ("quotes.xls", "excel"),
        ("quotes.xlsb", "excel"),
        ("quotes.xlsm", "excel"),
        ("notes.md", "unknown"),
        ("quotes.gz", "unknown"),
    ],
)
def test_file_kind(name: str, kind: str) -> None:
    assert file_kind(Path(name)) == kind


def test_text_suffixes() -> None:
    assert TEXT_SUFFIXES == {".csv", ".txt", ".tsv", ".dat"}


def test_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    expected = _quotes_frame()
    _write_delimited(path, expected, ",")
    assert sniff_delimiter(path) == ","
    assert peek_columns(path) == list(expected.columns)
    result = read_table(path)
    pd.testing.assert_frame_equal(result, expected)


def test_tsv_sniff_and_read(tmp_path: Path) -> None:
    path = tmp_path / "quotes.tsv"
    expected = _quotes_frame()
    _write_delimited(path, expected, "\t")
    assert sniff_delimiter(path) == "\t"
    result = read_table(path)
    pd.testing.assert_frame_equal(result, expected)


def test_semicolon_sniff_and_read(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    expected = _quotes_frame()
    _write_delimited(path, expected, ";")
    assert sniff_delimiter(path) == ";"
    result = read_table(path)
    pd.testing.assert_frame_equal(result, expected)


def test_gzip_text(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv.gz"
    expected = _quotes_frame()
    expected.to_csv(path, index=False, compression="gzip")
    assert file_kind(path) == "text_gz"
    assert sniff_delimiter(path) == ","
    assert peek_columns(path) == list(expected.columns)
    pd.testing.assert_frame_equal(read_table(path), expected)


def test_zip_text(tmp_path: Path) -> None:
    expected = _quotes_frame()
    path = tmp_path / "quotes.csv.zip"
    payload = expected.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("quotes.csv", payload)
    assert file_kind(path) == "text_gz"
    assert sniff_delimiter(path) == ","
    pd.testing.assert_frame_equal(read_table(path), expected)


def test_bom_header_names(tmp_path: Path) -> None:
    path = tmp_path / "spot.csv"
    path.write_bytes(b'\xef\xbb\xbf"Date","Price"\n"01/02/2013","1,462.40"\n')
    assert peek_columns(path)[0] == "Date"
    assert sniff_delimiter(path) == ","
    frame = read_table(path)
    assert list(frame.columns) == ["Date", "Price"]


def test_gzip_bom_header(tmp_path: Path) -> None:
    path = tmp_path / "spot.csv.gz"
    body = b'\xef\xbb\xbf"Date","Price"\n"01/02/2013","1,462.40"\n'
    with gzip.open(path, "wb") as handle:
        handle.write(body)
    assert peek_columns(path)[0] == "Date"


def test_usecols_matches_original_names(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    expected = _quotes_frame()
    _write_delimited(path, expected, ",")
    result = read_table(path, usecols=["date", "bid"])
    pd.testing.assert_frame_equal(result, expected.loc[:, ["date", "bid"]])


def test_usecols_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    _write_delimited(path, _quotes_frame(), ",")
    with pytest.raises(SchemaError, match="not in quotes.csv"):
        read_table(path, usecols=["date", "missing"])


def test_iter_table_chunks(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    expected = _quotes_frame()
    _write_delimited(path, expected, ",")
    chunks = list(iter_table(path, chunksize=2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    combined = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected)


def test_iter_table_gzip(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv.gz"
    expected = _quotes_frame()
    expected.to_csv(path, index=False, compression="gzip")
    combined = pd.concat(list(iter_table(path, chunksize=3)), ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected)


def test_unknown_kind_raises() -> None:
    path = Path("notes.md")
    with pytest.raises(SchemaError, match="unsupported table file type"):
        peek_columns(path)
    with pytest.raises(SchemaError, match="unsupported table file type"):
        read_table(path)
    with pytest.raises(SchemaError, match="unsupported table file type"):
        next(iter_table(path))


def test_stata_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "quotes.dta"
    expected = _quotes_frame()
    expected.to_stata(path, write_index=False)
    assert file_kind(path) == "stata"
    assert peek_columns(path) == list(expected.columns)
    result = read_table(path)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    chunks = list(iter_table(path, chunksize=2))
    combined = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected, check_dtype=False)


def test_stata_usecols(tmp_path: Path) -> None:
    path = tmp_path / "quotes.dta"
    expected = _quotes_frame()
    expected.to_stata(path, write_index=False)
    result = read_table(path, usecols=["strike", "bid"])
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.loc[:, ["strike", "bid"]],
        check_dtype=False,
    )


def test_parquet_missing_dependency_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "quotes.parquet"
    path.write_bytes(b"PAR1")

    def raise_import(*args, **kwargs):
        raise ImportError("Missing optional dependency 'pyarrow'")

    monkeypatch.setattr(pd, "read_parquet", raise_import)
    with pytest.raises(ImportError, match=r"crashfrontier\[parquet\]"):
        read_table(path)

    def missing_parquet_module():
        raise ImportError(
            "reading parquet requires pyarrow: pip install \"crashfrontier[parquet] "
            "@ git+https://github.com/Michaelzh6552/crashfrontier.git\""
        )

    monkeypatch.setattr("crashfrontier.data.readers._parquet_module", missing_parquet_module)
    with pytest.raises(ImportError, match=r"crashfrontier\[parquet\] @ git\+"):
        peek_columns(path)


def test_excel_missing_dependency_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "quotes.xlsx"
    path.write_bytes(b"PK")

    def raise_import(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd, "read_excel", raise_import)
    with pytest.raises(ImportError, match=r"crashfrontier\[excel\]"):
        read_table(path)
    with pytest.raises(ImportError, match=r"crashfrontier\[excel\] @ git\+"):
        peek_columns(path)


@pytest.mark.parametrize(
    ("name", "package"),
    [
        ("quotes.xls", "xlrd"),
        ("quotes.xlsb", "pyxlsb"),
    ],
)
def test_legacy_excel_import_error_names_real_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, package: str
) -> None:
    path = tmp_path / name
    path.write_bytes(b"fake-excel")

    def missing_engine(module: str):
        raise ImportError(f"No module named {module}")

    monkeypatch.setattr("crashfrontier.data.readers._import_engine", missing_engine)
    for reader in (read_table, peek_columns, lambda item: next(iter_table(item))):
        with pytest.raises(ImportError) as excinfo:
            reader(path)
        message = str(excinfo.value)
        assert package in message
        assert 'pip install "crashfrontier[excel] @ git+' in message
        assert f"openpyxl cannot read {path.suffix}" in message


def test_legacy_excel_read_error_is_not_blamed_on_openpyxl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "quotes.xls"
    path.write_bytes(b"fake-excel")
    monkeypatch.setattr("crashfrontier.data.readers._import_engine", lambda module: None)

    def raise_import(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd, "read_excel", raise_import)
    with pytest.raises(ImportError, match="xlrd") as excinfo:
        read_table(path)
    message = str(excinfo.value)
    assert "openpyxl cannot read .xls" in message
    assert 'pip install "crashfrontier[excel] @ git+' in message


def test_excel_reader_selects_engine_for_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engines: dict[str, str] = {}

    def fake_read_excel(path, **kwargs):
        engines[Path(path).suffix.lower()] = kwargs["engine"]
        return _quotes_frame()

    monkeypatch.setattr("crashfrontier.data.readers._import_engine", lambda module: None)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    for name, engine in (
        ("quotes.xlsx", "openpyxl"),
        ("quotes.xlsm", "openpyxl"),
        ("quotes.xls", "xlrd"),
        ("quotes.xlsb", "pyxlsb"),
    ):
        path = tmp_path / name
        path.write_bytes(b"fake-excel")
        frame = read_table(path)
        assert engines[path.suffix.lower()] == engine
        assert list(frame.columns) == list(_quotes_frame().columns)


def test_sas_read_uses_format_and_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "quotes.sas7bdat"
    path.write_bytes(b"fake-sas")
    expected = _quotes_frame()
    captured: dict[str, object] = {}

    def fake_read_sas(filepath, **kwargs):
        captured.update(kwargs)
        captured["filepath"] = filepath
        if kwargs.get("iterator") or kwargs.get("chunksize"):
            class Reader:
                column_names = list(expected.columns)

                def close(self) -> None:
                    return None

                def __iter__(self):
                    yield expected.iloc[:2]
                    yield expected.iloc[2:]

            return Reader()
        return expected.copy()

    monkeypatch.setattr(pd, "read_sas", fake_read_sas)
    assert peek_columns(path) == list(expected.columns)
    result = read_table(path)
    assert captured["format"] == "sas7bdat"
    assert captured["encoding"] == "latin-1"
    pd.testing.assert_frame_equal(result, expected)
    chunks = list(iter_table(path, chunksize=2))
    combined = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected)


def test_sniff_falls_back_to_comma(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("value\nonly\n", encoding="utf-8")
    assert sniff_delimiter(path) == ","


def test_parquet_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "quotes.parquet"
    expected = _quotes_frame()
    expected.to_parquet(path, index=False)
    assert peek_columns(path) == list(expected.columns)
    pd.testing.assert_frame_equal(read_table(path), expected)
    combined = pd.concat(list(iter_table(path)), ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected)


def test_excel_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "quotes.xlsx"
    expected = _quotes_frame()
    expected.to_excel(path, index=False)
    assert peek_columns(path) == list(expected.columns)
    pd.testing.assert_frame_equal(read_table(path), expected)
    chunks = list(iter_table(path, chunksize=2))
    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], expected)
