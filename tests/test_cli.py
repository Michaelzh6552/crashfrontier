from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crashfrontier.cli import _index_cache_names, _index_file_state, _read_dates_file, main
from crashfrontier.data.index import build_date_index


def _run(raw_dir: Path, cache: Path, command: str, *args: str) -> int:
    argv = [command, "--data", str(raw_dir), "--cache", str(cache), *args]
    return main(argv)


def test_inspect_prints_describe(raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = tmp_path / "cache"
    code = _run(raw_dir, cache, "inspect")
    assert code == 0
    out = capsys.readouterr().out
    for key in (
        "options.files",
        "options.kind",
        "date_format",
        "strike_scale",
        "forwards",
        "rates",
        "spot",
        "quote_filter.min_volume",
        "cache_dir",
    ):
        assert f"{key}:" in out
    assert "2021-07-21" in out
    assert "2021-07-28" in out


def test_inspect_column_overrides_and_options_paths(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    code = main(
        [
            "inspect",
            "--options",
            str(raw_dir / "option_price.csv"),
            "--forwards",
            str(raw_dir / "forward.csv"),
            "--rates",
            str(raw_dir / "DGS1MO.csv"),
            "--spot",
            str(raw_dir / "SPX.csv"),
            "--column",
            "date=date",
            "--column",
            "strike=strike_price",
            "--cache",
            str(cache),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "options.columns.date: date" in out
    assert "options.columns.strike_price: strike" in out
    assert "spot: True" in out


def test_inspect_accepts_an_options_directory(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "inspect",
            "--options",
            str(raw_dir),
            "--forwards",
            str(raw_dir / "forward.csv"),
            "--rates",
            str(raw_dir / "DGS1MO.csv"),
            "--spot",
            str(raw_dir / "SPX.csv"),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "options.files:" in out
    assert "option_price.csv" in out
    assert f"options.files: [{raw_dir}]" not in out


def test_read_dates_file_without_header_keeps_first_date(tmp_path: Path) -> None:
    path = tmp_path / "dates.txt"
    path.write_text(
        "2021-07-21\n2021-07-22\n2021-07-28\n",
        encoding="utf-8",
    )
    assert _read_dates_file(path, dayfirst=None) == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]


def test_read_dates_file_with_header_keeps_every_date(tmp_path: Path) -> None:
    path = tmp_path / "dates.csv"
    path.write_text("date\n2021-07-21\n2021-07-22\n2021-07-28\n", encoding="utf-8")
    assert _read_dates_file(path, dayfirst=None) == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]


def test_cli_formats_import_errors_without_traceback(
    raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(_args):
        raise ImportError("optional dependency is missing")

    monkeypatch.setattr("crashfrontier.cli._cmd_inspect", boom)
    code = _run(raw_dir, tmp_path / "cache", "inspect")
    assert code == 1
    err = capsys.readouterr().err
    assert "error: optional dependency is missing" in err
    assert "Traceback" not in err


def test_inspect_does_not_build_index(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"

    def boom(*_args, **_kwargs):
        raise AssertionError("CLI inspect must not build a date index")

    monkeypatch.setattr("crashfrontier.data.index.build_date_index", boom)
    monkeypatch.setattr("crashfrontier.cli.build_date_index", boom)
    assert _run(raw_dir, cache, "inspect") == 0
    assert not list(cache.glob("*.index.json"))


def test_inspect_hints_when_large_file_index_missing(
    raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr("crashfrontier.cli.DEFAULT_MEMORY_LIMIT", 0)

    def boom(*_args, **_kwargs):
        raise AssertionError("missing-index inspect must not scan the option file")

    monkeypatch.setattr("crashfrontier.data.index.build_date_index", boom)
    monkeypatch.setattr("crashfrontier.cli.build_date_index", boom)
    code = _run(raw_dir, cache, "inspect")
    assert code == 0
    out = capsys.readouterr().out
    assert "run crashfrontier index" in out
    assert "options.files:" in out
    assert not list(cache.glob("*.index.json"))


def test_index_force_rebuilds(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    option_path = raw_dir / "option_price.csv"
    index_path = cache / _index_cache_names([option_path])[option_path]
    assert _run(raw_dir, cache, "index") == 0
    assert index_path.is_file()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(payload["ranges"]) == 3
    first = capsys.readouterr().out
    assert "wrote" in first
    calls = {"n": 0}
    real = build_date_index

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.cli.build_date_index", wrapped)
    assert _run(raw_dir, cache, "index") == 0
    assert calls["n"] == 0
    skipped = capsys.readouterr().out
    assert "already current" in skipped
    assert _run(raw_dir, cache, "index", "--force") == 0
    assert calls["n"] == 1
    forced = capsys.readouterr().out
    assert "wrote" in forced


def test_expiries_iso_and_us_dates(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    assert _run(raw_dir, cache, "expiries", "2021-07-21") == 0
    iso = capsys.readouterr().out
    assert "expiry" in iso
    assert "2021-08-04" in iso
    assert _run(raw_dir, cache, "expiries", "07/21/2021") == 0
    us = capsys.readouterr().out
    assert "2021-08-04" in us


def test_expiries_missing_index_errors_without_scan(
    raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr("crashfrontier.cli.DEFAULT_MEMORY_LIMIT", 0)

    def boom(*_args, **_kwargs):
        raise AssertionError("expiries must not build a date index")

    monkeypatch.setattr("crashfrontier.data.index.build_date_index", boom)
    monkeypatch.setattr("crashfrontier.cli.build_date_index", boom)
    code = _run(raw_dir, cache, "expiries", "2021-07-21")
    assert code == 1
    err = capsys.readouterr().err
    assert any(line.startswith("error:") for line in err.splitlines())
    assert "run crashfrontier index" in err
    assert "Traceback" not in err


def test_error_exit_without_traceback(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    code = _run(raw_dir, cache, "expiries", "1999-01-01")
    assert code == 1
    captured = capsys.readouterr()
    assert any(line.startswith("error:") for line in captured.err.splitlines())
    assert "Traceback" not in captured.err


def test_debug_log_level_prints_traceback(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    code = main(
        [
            "expiries",
            "--data",
            str(raw_dir),
            "--cache",
            str(cache),
            "--log-level",
            "DEBUG",
            "1999-01-01",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert any(line.startswith("error:") for line in err.splitlines())
    assert "Traceback" in err


def test_frontier_nested_export_and_single_report(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    stem = tmp_path / "nested"
    code = _run(
        raw_dir,
        cache,
        "frontier",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--K",
        "0.9",
        "--nested",
        "--out",
        str(stem),
        "--export",
        "csv,json",
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "quote_set=put_pair" in out
    assert "quote_set=local_eight" in out
    assert "quote_set=put_wing" in out
    for name in ("summary", "vertices", "halfplanes", "quotes"):
        path = Path(f"{stem}_{name}.csv")
        assert path.is_file()
        table = pd.read_csv(path)
        assert not table.empty
    payload = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
    assert set(payload["frontiers"]) == {"put_pair", "local_eight", "put_wing"}
    assert Path(f"{stem}.pdf").is_file()
    assert Path(f"{stem}.png").is_file()

    report = tmp_path / "report"
    code = _run(
        raw_dir,
        cache,
        "frontier",
        "--date",
        "07/21/2021",
        "--target",
        "14",
        "--window",
        "7",
        "21",
        "--layout",
        "report",
        "--out",
        str(report),
    )
    assert code == 0
    report_out = capsys.readouterr().out
    assert "quote_set=" in report_out
    assert Path(f"{report}.pdf").is_file()
    assert Path(f"{report}.png").is_file()


def test_frontier_gamma_two_nested_and_report(
    raw_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crashfrontier.config import GridSettings as RealGrid

    monkeypatch.setattr(
        "crashfrontier.frontier.GridSettings",
        lambda: RealGrid(levels=(8,), max_normalized_gap=0.08),
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("gamma>0 report must not call binding_quotes")

    monkeypatch.setattr("crashfrontier.cli.binding_quotes", boom)
    cache = tmp_path / "cache"
    nested = tmp_path / "nested2"
    code = _run(
        raw_dir,
        cache,
        "frontier",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--K",
        "0.9",
        "--gamma",
        "2",
        "--nested",
        "--out",
        str(nested),
        "--export",
        "csv,json",
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "quote_set=put_pair" in out
    assert "quote_set=local_eight" in out
    assert "quote_set=put_wing" in out
    assert "quote_set=full_otm" in out
    assert "γ=2" in out
    assert Path(f"{nested}.pdf").is_file()
    assert Path(f"{nested}.png").is_file()
    payload = json.loads(Path(f"{nested}.json").read_text(encoding="utf-8"))
    assert set(payload["frontiers"]) == {
        "put_pair",
        "local_eight",
        "put_wing",
        "full_otm",
    }

    report = tmp_path / "report2"
    code = _run(
        raw_dir,
        cache,
        "frontier",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--gamma",
        "2",
        "--quote-set",
        "put_wing",
        "--layout",
        "report",
        "--out",
        str(report),
    )
    assert code == 0
    report_out = capsys.readouterr().out
    assert "quote_set=put_wing" in report_out
    assert Path(f"{report}.pdf").is_file()
    assert Path(f"{report}.png").is_file()


def test_frontier_support_cap_none(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crashfrontier.config import GridSettings as RealGrid

    monkeypatch.setattr(
        "crashfrontier.frontier.GridSettings",
        lambda: RealGrid(levels=(8,), max_normalized_gap=0.08),
    )
    cache = tmp_path / "cache"
    code = _run(
        raw_dir,
        cache,
        "frontier",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--gamma",
        "1",
        "--quote-set",
        "put_pair",
        "--support-cap",
        "none",
        "--out",
        str(tmp_path / "capnone"),
    )
    assert code == 0


def test_scenario_and_binding(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    code = _run(
        raw_dir,
        cache,
        "scenario",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--prob",
        "0.02",
        "--loss",
        "0.004",
    )
    assert code == 0
    scenario = capsys.readouterr().out
    assert "feasible:" in scenario
    assert "geometry:" in scenario
    assert "method:" in scenario

    stem = tmp_path / "bind"
    code = _run(
        raw_dir,
        cache,
        "binding",
        "--date",
        "2021-07-21",
        "--days",
        "14",
        "--direction",
        "1,0",
        "--out",
        str(stem),
    )
    assert code == 0
    binding = capsys.readouterr().out
    assert "kind" in binding
    assert "weight" in binding
    assert Path(f"{stem}.pdf").is_file()
    assert Path(f"{stem}.png").is_file()


def test_term_and_series_commands(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    term_stem = tmp_path / "term"
    code = _run(
        raw_dir,
        cache,
        "term",
        "--date",
        "2021-07-21",
        "--out",
        str(term_stem),
        "--export",
        "csv",
    )
    assert code == 0
    term_out = capsys.readouterr().out
    assert "target_days" in term_out
    table = pd.read_csv(f"{term_stem}.csv")
    assert "error" in table.columns
    success = table["error"].fillna("").astype(str).str.strip().eq("")
    assert int(success.sum()) == 3
    assert Path(f"{term_stem}.pdf").is_file()
    assert Path(f"{term_stem}.png").is_file()

    series_stem = tmp_path / "series"
    overlay = tmp_path / "overlay.csv"
    overlay.write_text("date,close\n2021-07-21,4300\n2021-07-22,4310\n2021-07-28,4290\n")
    dates_file = tmp_path / "dates.csv"
    dates_file.write_text("date\n2021-07-21\n2021-07-28\n")
    code = _run(
        raw_dir,
        cache,
        "series",
        "--dates",
        str(dates_file),
        "--target",
        "14",
        "--window",
        "7",
        "21",
        "--overlay",
        str(overlay),
        "--out",
        str(series_stem),
    )
    assert code == 0
    series_out = capsys.readouterr().out
    assert "requested_date" in series_out
    series_table = pd.read_csv(f"{series_stem}.csv")
    assert "requested_date" in series_table.columns
    assert "error" in series_table.columns
    assert len(series_table) == 2
    for value in ("prob", "loss", "depth"):
        assert Path(f"{series_stem}_{value}.pdf").is_file()
        assert Path(f"{series_stem}_{value}.png").is_file()

    daily = tmp_path / "daily"
    code = _run(
        raw_dir,
        cache,
        "series",
        "--start",
        "2021-07-21",
        "--end",
        "2021-07-28",
        "--freq",
        "D",
        "--out",
        str(daily),
    )
    assert code == 0
    daily_table = pd.read_csv(f"{daily}.csv")
    assert len(daily_table) == 3


def test_series_resume_is_not_global_cache(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crashfrontier import cli as cli_mod

    seen: dict[str, object] = {}
    real = cli_mod.frontier_series

    def wrapped(ds, *args, **kwargs):
        seen["cache_path"] = kwargs.get("cache_path")
        seen["dataset_cache"] = ds.describe()["cache_dir"]
        return real(ds, *args, **kwargs)

    monkeypatch.setattr("crashfrontier.cli.frontier_series", wrapped)
    index_dir = tmp_path / "index_cache"
    resume = tmp_path / "series_resume.csv"
    out = tmp_path / "series_out"
    code = main(
        [
            "series",
            "--data",
            str(raw_dir),
            "--cache",
            str(index_dir),
            "--resume",
            str(resume),
            "--start",
            "2021-07-21",
            "--end",
            "2021-07-28",
            "--freq",
            "D",
            "--target",
            "14",
            "--window",
            "7",
            "21",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert Path(seen["cache_path"]) == resume
    assert Path(str(seen["dataset_cache"])) == index_dir
    assert resume.is_file()
    assert Path(f"{out}.csv").is_file()


def test_shared_stem_index_names_do_not_collide(tmp_path: Path) -> None:
    left = tmp_path / "one" / "option_price.csv"
    right = tmp_path / "two" / "option_price.csv"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("date\n", encoding="utf-8")
    right.write_text("date\n", encoding="utf-8")
    names = _index_cache_names([left, right])
    assert names[left] != names[right]
    assert names[left] != "option_price.index.json"
    assert names[right] != "option_price.index.json"


def test_index_cache_names_follow_data_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "option_price.csv"
    path.write_text("date\n", encoding="utf-8")

    def fake(paths, suffix):
        assert suffix == "index.json"
        return {paths[0]: "from-data.index.json"}

    monkeypatch.setattr("crashfrontier.data.source._cache_filenames", fake)
    assert _index_cache_names([path])[path] == "from-data.index.json"


def test_index_file_state_checks_source_and_dayfirst(tmp_path: Path) -> None:
    csv_path = tmp_path / "option_price.csv"
    csv_path.write_text("date,strike\n2021-07-21,1\n", encoding="utf-8")
    info = csv_path.stat()
    index_path = tmp_path / "idx.json"
    payload = {
        "source": str(csv_path.resolve()),
        "size": info.st_size,
        "mtime": info.st_mtime,
        "date_column": 0,
        "delimiter": ",",
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _index_file_state(csv_path, index_path, 0, ",", dayfirst=None) == "ok"
    payload["dayfirst"] = True
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _index_file_state(csv_path, index_path, 0, ",", dayfirst=None) == "stale"
    assert _index_file_state(csv_path, index_path, 0, ",", dayfirst=True) == "ok"
    payload["source"] = str((tmp_path / "other.csv").resolve())
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _index_file_state(csv_path, index_path, 0, ",", dayfirst=True) == "stale"
    payload.pop("source")
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _index_file_state(csv_path, index_path, 0, ",", dayfirst=True) == "missing"
