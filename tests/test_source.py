from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from crashfrontier.data.index import (
    DateIndex,
    build_date_index,
    load_date_index,
    read_rows_by_ranges,
)
from crashfrontier.data.readers import peek_columns
from crashfrontier.data.source import (
    ChunkScanSource,
    FrameSource,
    IndexedTextSource,
    ParquetSource,
    classify_table,
    open_option_source,
)
from crashfrontier.errors import NoSuchDateError, SchemaError
from tests.synthetic import write_optionmetrics_dir


def _resolved_cache_token(path: Path) -> str:
    canonical = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _data_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        handle.readline()
        return sum(1 for _ in handle)


def _open_indexed(options, tmp_path: Path, **kwargs):
    cache_dir = kwargs.pop("cache_dir", tmp_path / "cache")
    return open_option_source(
        options, cache_dir=cache_dir, memory_limit=0, **kwargs
    )


def test_indexed_text_covers_dates_and_row_count(raw_dir: Path, tmp_path: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    source, mapping = _open_indexed(option_path, tmp_path)
    assert isinstance(source, IndexedTextSource)
    assert mapping.date == "date"
    dates = source.dates()
    assert dates == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    expected = pd.read_csv(option_path)
    total = sum(len(source.rows(date)) for date in dates)
    assert total == len(expected) == _data_row_count(option_path)
    index_path = tmp_path / "cache" / "option_price.index.json"
    index = load_date_index(
        option_path, index_path, date_column=1, delimiter=","
    )
    stored = json.loads(index_path.read_text(encoding="utf-8"))
    assert Path(stored["source"]).resolve() == option_path.resolve()
    assert set(index.ranges) == set(expected["date"].astype(str))
    reconstructed = pd.concat(
        [read_rows_by_ranges(index, key, None) for key in index.ranges],
        ignore_index=True,
    )
    assert len(reconstructed) == len(expected)


def test_indexed_text_rows_keep_original_columns(raw_dir: Path, tmp_path: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    source, _ = _open_indexed(option_path, tmp_path)
    rows = source.rows("2021-07-21")
    assert list(rows.columns) == peek_columns(option_path) == source.columns
    assert rows["date"].astype(str).str.startswith("2021-07-21").all()
    assert len(rows) == 200
    assert "best_bid" in rows.columns
    assert "_date" not in rows.columns


def test_mtime_change_rebuilds_index(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    option_path = raw_dir / "option_price.csv"
    cache_dir = tmp_path / "cache"
    _open_indexed(option_path, tmp_path, cache_dir=cache_dir)
    index_path = cache_dir / "option_price.index.json"
    original = json.loads(index_path.read_text(encoding="utf-8"))
    calls = {"n": 0}
    real = build_date_index

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.data.index.build_date_index", wrapped)
    _open_indexed(option_path, tmp_path, cache_dir=cache_dir)
    assert calls["n"] == 0

    stat = option_path.stat()
    os.utime(option_path, (stat.st_mtime + 10, stat.st_mtime + 10))
    _open_indexed(option_path, tmp_path, cache_dir=cache_dir)
    assert calls["n"] == 1
    rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
    assert rebuilt["mtime"] != original["mtime"]
    assert rebuilt["size"] == original["size"]
    assert rebuilt["date_column"] == 1


def test_frame_source_from_dataframe_and_small_file(raw_dir: Path, tmp_path: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    frame = pd.read_csv(option_path)
    source, mapping = open_option_source(frame, cache_dir=tmp_path / "cache")
    assert isinstance(source, FrameSource)
    assert mapping.date == "date"
    assert source.dates()[0] == pd.Timestamp("2021-07-21")
    rows = source.rows(pd.Timestamp("2021-07-22"))
    assert len(rows) == 200
    assert list(rows.columns) == list(frame.columns)
    with pytest.raises(ValueError, match="in-memory"):
        source.spec()

    file_source, _ = open_option_source(option_path, cache_dir=tmp_path / "cache")
    assert isinstance(file_source, FrameSource)
    pd.testing.assert_frame_equal(
        file_source.rows("2021-07-21").reset_index(drop=True),
        source.rows("2021-07-21").reset_index(drop=True),
    )


def test_chunk_scan_source_for_gzip(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    variant = write_optionmetrics_dir(tmp_path / "gz", name="yyyymmdd_gz_column_forward")
    gz_path = variant / "option_price.csv.gz"
    with caplog.at_level(logging.WARNING, logger="crashfrontier"):
        source, mapping = _open_indexed(gz_path, tmp_path)
    assert isinstance(source, ChunkScanSource)
    assert mapping.date == "date"
    assert "chunked scans reread the entire compressed file" in caplog.text
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    rows = source.rows("2021-07-21")
    assert len(rows) == 200
    assert mapping.date in rows.columns
    cache_path = tmp_path / "cache" / "option_price.csv.dates.json"
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["size"] == gz_path.stat().st_size
    assert payload["dayfirst"] is None
    assert Path(payload["source"]).resolve() == gz_path.resolve()
    spec = source.spec()
    json.dumps(spec)
    assert spec["kind"] == "chunk_scan"
    assert spec["paths"] == [str(gz_path)]


def test_multi_file_merge_from_directory_and_list(tmp_path: Path) -> None:
    variant = write_optionmetrics_dir(tmp_path / "split", name="date9_split_dollar_novolume")
    source, _ = _open_indexed(variant, tmp_path)
    assert isinstance(source, IndexedTextSource)
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    first = source.rows("2021-07-21")
    last = source.rows("2021-07-28")
    assert not first.empty
    assert not last.empty
    a_rows = _data_row_count(variant / "opt_2021a.csv")
    b_rows = _data_row_count(variant / "opt_2021b.csv")
    assert sum(len(source.rows(date)) for date in source.dates()) == a_rows + b_rows

    listed, _ = _open_indexed(
        [variant / "opt_2021a.csv", variant / "opt_2021b.csv"], tmp_path / "listed"
    )
    assert listed.dates() == source.dates()
    spec = source.spec()
    json.dumps(spec)
    assert spec["kind"] == "indexed_text"
    assert spec["memory_limit"] == 0
    assert {Path(path).name for path in spec["paths"]} == {"opt_2021a.csv", "opt_2021b.csv"}


def test_same_named_files_do_not_share_date_indexes(tmp_path: Path) -> None:
    header = "date,expiry,cp_flag,strike,bid,ask,volume\n"
    first = tmp_path / "a" / "options.csv"
    second = tmp_path / "b" / "options.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        header + "2021-07-21,2021-08-20,P,90,1,2,10\n",
        encoding="utf-8",
    )
    second.write_text(
        header + "2021-07-22,2021-08-20,C,90,1,2,10\n",
        encoding="utf-8",
    )
    stamp = 1_700_000_000
    os.utime(first, (stamp, stamp))
    os.utime(second, (stamp, stamp))
    source, _ = _open_indexed([first, second], tmp_path)
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
    ]
    assert source.rows("2021-07-21")["cp_flag"].tolist() == ["P"]
    assert source.rows("2021-07-22")["cp_flag"].tolist() == ["C"]
    cache = tmp_path / "cache"
    name_a = f"options-{_resolved_cache_token(first)}.index.json"
    name_b = f"options-{_resolved_cache_token(second)}.index.json"
    assert name_a != name_b
    assert (cache / name_a).is_file()
    assert (cache / name_b).is_file()
    assert not (cache / "options.index.json").exists()
    for path, name in ((first, name_a), (second, name_b)):
        payload = json.loads((cache / name).read_text(encoding="utf-8"))
        assert Path(payload["source"]).resolve() == path.resolve()


def test_chunk_date_cache_respects_dayfirst(tmp_path: Path) -> None:
    path = tmp_path / "options.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "date,expiry,cp_flag,strike,bid,ask,volume\n"
            "05/06/2021,20/06/2021,P,90,1,2,10\n"
        )
    cache = tmp_path / "cache"
    month_first, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=False)
    day_first, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=True)
    assert month_first.dates() == [pd.Timestamp("2021-05-06")]
    assert day_first.dates() == [pd.Timestamp("2021-06-05")]
    month_again, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=False)
    assert month_again.dates() == [pd.Timestamp("2021-05-06")]


def test_index_cache_rejects_different_source(tmp_path: Path) -> None:
    header = "date,expiry,cp_flag,strike,bid,ask,volume\n"
    first = tmp_path / "a" / "options.csv"
    second = tmp_path / "b" / "options.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(header + "2021-07-21,2021-08-20,P,90,1,2,10\n", encoding="utf-8")
    second.write_text(header + "2021-07-22,2021-08-20,C,90,1,2,10\n", encoding="utf-8")
    stamp = 1_700_000_000
    os.utime(first, (stamp, stamp))
    os.utime(second, (stamp, stamp))
    cache_path = tmp_path / "shared.index.json"
    built = build_date_index(first, cache_path, date_column=0, delimiter=",")
    assert "2021-07-21" in built.ranges
    loaded = load_date_index(second, cache_path, date_column=0, delimiter=",")
    assert set(loaded.ranges) == {"2021-07-22"}
    assert loaded.source.resolve() == second.resolve()
    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert Path(stored["source"]).resolve() == second.resolve()


def test_separate_same_stem_opens_do_not_share_index(tmp_path: Path) -> None:
    header = "date,expiry,cp_flag,strike,bid,ask,volume\n"
    first = tmp_path / "a" / "options.csv"
    second = tmp_path / "b" / "options.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(header + "2021-07-21,2021-08-20,P,90,1,2,10\n", encoding="utf-8")
    second.write_text(header + "2021-07-22,2021-08-20,C,90,1,2,10\n", encoding="utf-8")
    stamp = 1_700_000_000
    os.utime(first, (stamp, stamp))
    os.utime(second, (stamp, stamp))
    cache = tmp_path / "cache"
    first_source, _ = _open_indexed(first, tmp_path, cache_dir=cache)
    second_source, _ = _open_indexed(second, tmp_path, cache_dir=cache)
    assert first_source.dates() == [pd.Timestamp("2021-07-21")]
    assert second_source.rows("2021-07-22")["cp_flag"].tolist() == ["C"]
    again, _ = _open_indexed(first, tmp_path, cache_dir=cache)
    assert again.rows("2021-07-21")["cp_flag"].tolist() == ["P"]


def test_dates_cache_rejects_foreign_source_and_dayfirst(tmp_path: Path) -> None:
    path = tmp_path / "options.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "date,expiry,cp_flag,strike,bid,ask,volume\n"
            "05/06/2021,20/06/2021,P,90,1,2,10\n"
        )
    cache = tmp_path / "cache"
    loaded, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=False)
    assert loaded.dates() == [pd.Timestamp("2021-05-06")]
    cache_path = cache / "options.csv.dates.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["dates"] = ["1999-01-01"]
    payload["dayfirst"] = True
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    month_first, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=False)
    assert month_first.dates() == [pd.Timestamp("2021-05-06")]

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["dates"] = ["1999-01-01"]
    payload["dayfirst"] = False
    payload["source"] = str((tmp_path / "other" / "options.csv.gz").resolve())
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    rescanned, _ = _open_indexed(path, tmp_path, cache_dir=cache, dayfirst=False)
    assert rescanned.dates() == [pd.Timestamp("2021-05-06")]
    restored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert Path(restored["source"]).resolve() == path.resolve()
    assert restored["dayfirst"] is False


def test_same_stem_dates_caches_use_resolved_path_hash(tmp_path: Path) -> None:
    header = "date,expiry,cp_flag,strike,bid,ask,volume\n"

    def write_gz(path: Path, row: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(header + row)

    first = tmp_path / "a" / "options.csv.gz"
    second = tmp_path / "b" / "options.csv.gz"
    write_gz(first, "2021-07-21,2021-08-20,P,90,1,2,10\n")
    write_gz(second, "2021-07-22,2021-08-20,C,90,1,2,10\n")
    stamp = 1_700_000_000
    os.utime(first, (stamp, stamp))
    os.utime(second, (stamp, stamp))
    source, _ = _open_indexed([first, second], tmp_path)
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
    ]
    assert source.rows("2021-07-21")["cp_flag"].tolist() == ["P"]
    cache = tmp_path / "cache"
    name_a = f"options.csv-{_resolved_cache_token(first)}.dates.json"
    name_b = f"options.csv-{_resolved_cache_token(second)}.dates.json"
    assert (cache / name_a).is_file()
    assert (cache / name_b).is_file()
    assert not (cache / "options.csv.dates.json").exists()


def test_option_rows_drop_timestamp_timezone(raw_dir: Path, tmp_path: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    frame = pd.read_csv(option_path)
    source, _ = open_option_source(frame)
    aware = pd.Timestamp("2021-07-22 18:00", tz="UTC")
    assert len(source.rows(aware)) == 200
    indexed, _ = _open_indexed(option_path, tmp_path)
    assert len(indexed.rows(pd.Timestamp("2021-07-21", tz="UTC"))) == 200


def test_directory_picks_option_files_only(raw_dir: Path, tmp_path: Path) -> None:
    source, _ = _open_indexed(raw_dir, tmp_path)
    assert isinstance(source, IndexedTextSource)
    assert source.spec()["paths"] == [str(raw_dir / "option_price.csv")]
    assert classify_table(raw_dir / "option_price.csv") == "options"
    assert classify_table(raw_dir / "forward.csv") == "forwards"
    assert classify_table(raw_dir / "DGS1MO.csv") == "rate_series"
    assert classify_table(raw_dir / "SPX.csv") == "spot"
    assert classify_table(raw_dir / "VIX_History.csv") == "spot"


def test_classify_table_zero_curve_and_unknown(tmp_path: Path) -> None:
    variant = write_optionmetrics_dir(tmp_path / "gz", name="yyyymmdd_gz_column_forward")
    assert classify_table(variant / "zerocd.csv") == "zero_curve"
    notes = tmp_path / "notes.txt"
    notes.write_text("a,b\n1,2\n", encoding="utf-8")
    assert classify_table(notes) == "unknown"
    assert classify_table(tmp_path / "missing.md") == "unknown"


def test_open_option_source_column_mismatch(raw_dir: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.csv"
    pd.DataFrame({"date": ["2021-07-21"], "x": [1]}).to_csv(other, index=False)
    with pytest.raises(SchemaError, match="columns do not match"):
        _open_indexed([raw_dir / "option_price.csv", other], tmp_path)


def test_no_such_date_mentions_neighbors(raw_dir: Path, tmp_path: Path) -> None:
    source, _ = _open_indexed(raw_dir / "option_price.csv", tmp_path)
    with pytest.raises(NoSuchDateError, match="2021-07-23") as excinfo:
        source.rows("2021-07-23")
    message = str(excinfo.value)
    assert "2021-07-22" in message
    assert "2021-07-28" in message


def test_noncontiguous_date_ranges(tmp_path: Path) -> None:
    path = tmp_path / "shuffled.csv"
    path.write_text(
        "date,exdate,cp_flag,strike_price,best_bid,best_offer\n"
        "2021-07-21,2021-08-01,P,1000,1,2\n"
        "2021-07-22,2021-08-01,C,1000,1,2\n"
        "2021-07-21,2021-08-01,C,1100,1,2\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "shuffled.index.json"
    index = build_date_index(path, index_path, date_column=0, delimiter=",")
    assert isinstance(index, DateIndex)
    assert len(index.ranges["2021-07-21"]) == 2
    rows = read_rows_by_ranges(index, "2021-07-21", ["date", "cp_flag"])
    assert list(rows["cp_flag"]) == ["P", "C"]
    loaded = load_date_index(path, index_path, date_column=0, delimiter=",")
    assert loaded.ranges == index.ranges
    loaded_other = load_date_index(path, index_path, date_column=1, delimiter=",")
    assert loaded_other.date_column == 1
    assert set(loaded_other.ranges) == {"2021-08-01"}


def test_indexed_source_spec_is_jsonable(raw_dir: Path, tmp_path: Path) -> None:
    source, mapping = _open_indexed(raw_dir / "option_price.csv", tmp_path)
    payload = json.dumps(source.spec())
    spec = json.loads(payload)
    assert spec["kind"] == "indexed_text"
    assert spec["column_map"]["date"] == mapping.date
    assert spec["dayfirst"] is None


def test_parquet_source(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    variant = write_optionmetrics_dir(tmp_path / "parquet", name="parquet")
    source, mapping = _open_indexed(variant, tmp_path)
    assert isinstance(source, ParquetSource)
    assert mapping.date == "date"
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    rows = source.rows("2021-07-21")
    assert len(rows) == 200
    assert "strike_price" in rows.columns
    spec = source.spec()
    json.dumps(spec)
    assert spec["kind"] == "parquet"


def test_parquet_source_keeps_all_times_on_one_date(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "options.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-07-21 10:00", "2021-07-21 16:00"]),
            "expiry": pd.to_datetime(["2021-08-20", "2021-08-20"]),
            "cp_flag": ["P", "C"],
            "strike": [90.0, 110.0],
            "bid": [1.0, 1.0],
            "ask": [2.0, 2.0],
            "volume": [10, 10],
        }
    ).to_parquet(path)
    source, _ = _open_indexed(path, tmp_path)
    rows = source.rows("2021-07-21")
    assert len(rows) == 2
    assert set(rows["cp_flag"]) == {"P", "C"}
    aware = source.rows(pd.Timestamp("2021-07-21 22:00", tz="UTC"))
    assert len(aware) == 2


def test_uppercase_semicolon_indexed(tmp_path: Path) -> None:
    variant = write_optionmetrics_dir(tmp_path / "semi", name="uppercase_semicolon_mdy")
    source, mapping = _open_indexed(variant, tmp_path)
    assert isinstance(source, IndexedTextSource)
    assert mapping.date == "DATE"
    assert source.dates() == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    rows = source.rows("2021-07-21")
    assert mapping.date in rows.columns
    assert len(rows) == 200
