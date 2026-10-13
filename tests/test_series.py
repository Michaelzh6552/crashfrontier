from __future__ import annotations

import inspect
import logging
import pickle
from pathlib import Path

import pandas as pd
import pytest

from crashfrontier.data.dataset import Dataset
from crashfrontier.errors import InfeasibleCrossSectionError, NoSuchDateError
from crashfrontier.series import (
    _SUMMARY_COLUMNS,
    _init_series_worker,
    _series_worker,
    frontier_series,
    pick_dates,
    term_structure,
    threshold_profile,
)
from tests.synthetic import FrameBundle, write_optionmetrics_dir
from tests.test_frontier import THRESHOLD, _quotes, _section


def _ok(table: pd.DataFrame) -> pd.Series:
    return table["error"].fillna("").astype(str).str.strip().eq("")


def test_pick_dates_four_frequencies() -> None:
    available = pd.to_datetime(
        [
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-08",
            "2021-01-11",
            "2021-01-13",
            "2021-01-29",
            "2021-02-26",
            "2021-03-31",
        ]
    )
    daily = pick_dates(available, freq="D")
    weekly = pick_dates(available, freq="W")
    monthly = pick_dates(available, freq="M")
    quarterly = pick_dates(available, freq="Q")
    assert daily == [pd.Timestamp(item).normalize() for item in available]
    assert weekly == [
        pd.Timestamp("2021-01-06"),
        pd.Timestamp("2021-01-13"),
        pd.Timestamp("2021-01-29"),
        pd.Timestamp("2021-02-26"),
        pd.Timestamp("2021-03-31"),
    ]
    assert monthly == [
        pd.Timestamp("2021-01-29"),
        pd.Timestamp("2021-02-26"),
        pd.Timestamp("2021-03-31"),
    ]
    assert quarterly == [pd.Timestamp("2021-03-31")]
    clipped = pick_dates(available, start="2021-01-11", end="2021-02-28", freq="D")
    assert clipped[0] == pd.Timestamp("2021-01-11")
    assert clipped[-1] == pd.Timestamp("2021-02-26")
    with pytest.raises(ValueError, match="freq"):
        pick_dates(available, freq="Y")


def test_term_structure_three_tenors(dataset: Dataset) -> None:
    table, fronts = term_structure(dataset, "2021-07-21", tenors=(7, 14, 30))
    assert list(table.columns[: len(_SUMMARY_COLUMNS)]) == list(_SUMMARY_COLUMNS)
    assert list(table.columns[-2:]) == ["target_days", "error"]
    assert len(table) == 3
    assert _ok(table).sum() == 3
    assert set(fronts) == {7, 14, 30}
    assert list(table["target_days"]) == [7, 14, 30]
    assert list(table["days"]) == [7, 14, 30]


def test_term_structure_records_errors_and_duplicate_expiry(dataset: Dataset) -> None:
    table, fronts = term_structure(dataset, "2021-07-21")
    assert len(table) == 5
    assert _ok(table).sum() == 3
    assert len(fronts) == 3
    failed = table.loc[~_ok(table)]
    assert set(failed["target_days"]) == {60, 90}
    assert failed["error"].str.len().gt(0).all()

    dup, dup_fronts = term_structure(dataset, "2021-07-21", tenors=(7, 8))
    assert len(dup) == 2
    assert _ok(dup).sum() == 1
    kept = dup.loc[_ok(dup)].iloc[0]
    dropped = dup.loc[~_ok(dup)].iloc[0]
    assert int(kept["target_days"]) == 7
    assert int(dropped["target_days"]) == 8
    assert dropped["error"] == "expiry already used"
    assert set(dup_fronts) == {7}


def test_term_structure_item_error_does_not_abort(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise InfeasibleCrossSectionError("full_otm infeasible")

    monkeypatch.setattr("crashfrontier.series.compute_frontier", boom)
    table, fronts = term_structure(dataset, "2021-07-21", tenors=(7, 14), gamma=2)
    assert len(table) == 2
    assert fronts == {}
    assert table["error"].str.contains("infeasible").all()


def test_threshold_profile_on_synthetic_section() -> None:
    section = _section(_quotes())
    table, fronts = threshold_profile(section, thresholds=(0.85, THRESHOLD))
    assert list(table.columns[: len(_SUMMARY_COLUMNS)]) == list(_SUMMARY_COLUMNS)
    assert table.columns[-1] == "error"
    assert len(table) == 2
    assert _ok(table).all()
    assert set(fronts) == {0.85, THRESHOLD}
    assert list(table["K"]) == [0.85, THRESHOLD]


def test_frontier_series_workers_and_cache(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = dict(start="2021-07-21", end="2021-07-28", freq="D", target_days=14, window=(7, 21))
    one = frontier_series(dataset, workers=1, **common)
    two = frontier_series(dataset, workers=2, **common)
    assert len(one) == 3
    assert list(one.columns[: len(_SUMMARY_COLUMNS)]) == list(_SUMMARY_COLUMNS)
    assert list(one.columns[-2:]) == ["requested_date", "error"]
    assert _ok(one).all()
    left = one.drop(columns=["elapsed_seconds"])
    right = two.drop(columns=["elapsed_seconds"])
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-9, rtol=0)
    assert list(one["requested_date"]) == sorted(one["requested_date"])

    cache_path = tmp_path / "series.csv"
    calls = {"n": 0}
    real = __import__("crashfrontier.series", fromlist=["compute_frontier"]).compute_frontier

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.series.compute_frontier", wrapped)
    first = frontier_series(dataset, workers=1, cache_path=cache_path, **common)
    assert calls["n"] == 3
    assert cache_path.is_file()
    second = frontier_series(dataset, workers=1, cache_path=cache_path, **common)
    assert calls["n"] == 3
    pd.testing.assert_frame_equal(
        first.drop(columns=["elapsed_seconds"]),
        second.drop(columns=["elapsed_seconds"]),
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


def test_frontier_series_snap_and_on_error(dataset: Dataset) -> None:
    skipped = frontier_series(
        dataset,
        dates=["2021-07-23", "1999-01-01"],
        snap=None,
        on_error="skip",
        target_days=14,
        window=(7, 21),
    )
    assert len(skipped) == 2
    assert list(pd.to_datetime(skipped["requested_date"])) == [
        pd.Timestamp("1999-01-01"),
        pd.Timestamp("2021-07-23"),
    ]
    assert not _ok(skipped).any()

    snapped = frontier_series(
        dataset,
        dates=["2021-07-23"],
        snap="backward",
        target_days=14,
        window=(7, 21),
    )
    assert _ok(snapped).all()
    assert pd.Timestamp(snapped.iloc[0]["date"]) == pd.Timestamp("2021-07-22")
    assert pd.Timestamp(snapped.iloc[0]["requested_date"]) == pd.Timestamp("2021-07-23")

    forward = frontier_series(
        dataset,
        dates=["2021-07-23"],
        snap="forward",
        target_days=14,
        window=(7, 21),
    )
    assert pd.Timestamp(forward.iloc[0]["date"]) == pd.Timestamp("2021-07-28")

    with pytest.raises(NoSuchDateError):
        frontier_series(
            dataset,
            dates=["1999-01-01"],
            snap=None,
            on_error="raise",
            target_days=14,
            window=(7, 21),
        )


def test_memory_source_falls_back_to_one_worker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bundle = write_optionmetrics_dir(tmp_path / "frames", name="frames")
    assert isinstance(bundle, FrameBundle)
    dataset = Dataset.from_frames(
        bundle.options,
        forwards=bundle.forwards,
        rates=bundle.rates,
        spot=bundle.spot,
        cache_dir=tmp_path / "cache",
    )
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    table = frontier_series(
        dataset,
        freq="D",
        workers=2,
        target_days=14,
        window=(7, 21),
    )
    assert "in-memory" in caplog.text
    assert "workers=1" in caplog.text
    assert len(table) == 3
    assert _ok(table).all()


def test_worker_is_module_top_level_and_pickleable(dataset: Dataset) -> None:
    assert inspect.getmodule(_series_worker).__name__ == "crashfrontier.series"
    assert inspect.getmodule(_init_series_worker).__name__ == "crashfrontier.series"
    spec = dataset.spec()
    job = {
        "index": 0,
        "spec": spec,
        "requested": "2021-07-21",
        "K": 0.9,
        "gamma": 0,
        "quote_set": None,
        "days": None,
        "target_days": 14,
        "window": (7, 21),
        "am_settlement": None,
        "snap": "backward",
        "on_error": "skip",
        "frontier_kwargs": {},
    }
    pickle.loads(pickle.dumps(job))
    pickle.dumps(_series_worker)
    pickle.dumps(_init_series_worker)


def test_series_dates_drop_timezone(dataset: Dataset) -> None:
    chosen = pick_dates(
        [
            pd.Timestamp("2021-07-21", tz="UTC"),
            pd.Timestamp("2021-07-22", tz="Asia/Shanghai"),
        ],
        freq="D",
    )
    assert all(item.tzinfo is None for item in chosen)
    assert chosen == [pd.Timestamp("2021-07-21"), pd.Timestamp("2021-07-22")]
    table = frontier_series(
        dataset,
        dates=[pd.Timestamp("2021-07-21", tz="UTC")],
        target_days=14,
        window=(7, 21),
        snap=None,
    )
    assert table["requested_date"].dt.tz is None
    assert table["date"].dt.tz is None
    assert pd.Timestamp(table.iloc[0]["requested_date"]) == pd.Timestamp("2021-07-21")
    assert _ok(table).all()


def test_error_row_keeps_quote_set_and_strikes(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crashfrontier.data.quote_sets import select_quote_set

    def boom(*_args, **_kwargs):
        raise InfeasibleCrossSectionError("infeasible", quote_set="put_pair")

    monkeypatch.setattr("crashfrontier.series.compute_frontier", boom)
    table = frontier_series(
        dataset,
        dates=["2021-07-21"],
        quote_set="put_pair",
        target_days=14,
        window=(7, 21),
    )
    row = table.iloc[0]
    assert str(row["error"]).strip()
    assert row["quote_set"] == "put_pair"
    section = dataset.cross_section("2021-07-21", target_days=14, window=(7, 21))
    expected = int(
        select_quote_set(section.quotes, 0.9, "put_pair", "riskneutral")["strike"].nunique()
    )
    assert int(row["n_strikes"]) == expected


def test_resume_uses_signature_not_only_requested_date(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "series.csv"
    calls = {"n": 0}
    real = __import__("crashfrontier.series", fromlist=["compute_frontier"]).compute_frontier

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.series.compute_frontier", wrapped)
    common = dict(
        dates=["2021-07-21"],
        target_days=14,
        window=(7, 21),
        cache_path=cache,
    )
    first = frontier_series(dataset, **common)
    assert calls["n"] == 1
    assert "cache_signature" not in first.columns
    stored = pd.read_csv(cache)
    assert "cache_signature" in stored.columns
    assert stored["cache_signature"].notna().all()
    frontier_series(dataset, **common)
    assert calls["n"] == 1

    day_cache = tmp_path / "days.csv"
    frontier_series(
        dataset,
        dates=["2021-07-21"],
        days=14,
        target_days=30,
        window=(1, 5),
        cache_path=day_cache,
    )
    assert calls["n"] == 2
    frontier_series(
        dataset,
        dates=["2021-07-21"],
        days=14,
        target_days=7,
        window=(20, 40),
        cache_path=day_cache,
    )
    assert calls["n"] == 2

    frontier_series(dataset, K=0.85, **common)
    assert calls["n"] == 3
    first.to_csv(cache, index=False)
    frontier_series(dataset, **common)
    assert calls["n"] == 4


def test_resume_recomputes_error_rows(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "errors.csv"
    state = {"fail": True}
    calls = {"n": 0}
    real = __import__("crashfrontier.series", fromlist=["compute_frontier"]).compute_frontier

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if state["fail"]:
            raise InfeasibleCrossSectionError("nope", quote_set="put_wing")
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.series.compute_frontier", wrapped)
    kwargs = dict(
        dates=["2021-07-21"],
        target_days=14,
        window=(7, 21),
        cache_path=cache,
    )
    failed = frontier_series(dataset, **kwargs)
    assert calls["n"] == 1
    assert not _ok(failed).any()
    assert failed.iloc[0]["quote_set"] == "put_wing"
    state["fail"] = False
    recovered = frontier_series(dataset, **kwargs)
    assert calls["n"] == 2
    assert _ok(recovered).all()
