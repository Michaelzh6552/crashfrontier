from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crashfrontier.data import Dataset, classify_table
from crashfrontier.data.schema import CANONICAL_FIELDS
from crashfrontier.data.source import FrameSource
from crashfrontier.errors import NoEligibleExpiryError, NoSuchDateError, SchemaError
from tests.synthetic import FrameBundle, write_optionmetrics_dir

QUOTE_COMPARE_COLS = ["strike", "norm_l", "norm_u"]
EXPIRY_COLUMNS = [
    "expiry",
    "am_settlement",
    "root",
    "days",
    "n_rows",
    "n_put_strikes",
    "n_call_strikes",
    "forward",
    "forward_source",
]
DESCRIBE_KEYS = {
    "options",
    "date_format",
    "strike_scale",
    "forwards",
    "rates",
    "spot",
    "quote_filter",
    "cache_dir",
}
OPTION_DESCRIBE_KEYS = {"files", "kind", "n_dates", "first_date", "last_date", "columns"}
CANONICAL_ROW_COLUMNS = list(CANONICAL_FIELDS) + ["days"]


def _open_variant(variant: Path | FrameBundle, tmp_path: Path) -> Dataset:
    cache = tmp_path / "variant_cache"
    if isinstance(variant, FrameBundle):
        return Dataset.from_frames(
            variant.options,
            forwards=variant.forwards,
            rates=variant.rates,
            spot=variant.spot,
            cache_dir=cache,
        )
    return Dataset.from_directory(variant, cache_dir=cache)


def test_constructor_accepts_path_list_directory_and_frame(raw_dir: Path, tmp_path: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    forwards = raw_dir / "forward.csv"
    rates = raw_dir / "DGS1MO.csv"
    spot = raw_dir / "SPX.csv"
    common = {"forwards": forwards, "rates": rates, "spot": spot}
    from_path = Dataset(option_path, cache_dir=tmp_path / "c1", **common)
    from_list = Dataset([option_path], cache_dir=tmp_path / "c2", **common)
    from_dir = Dataset(raw_dir, cache_dir=tmp_path / "c3", **common)
    from_frame = Dataset(
        pd.read_csv(option_path),
        forwards=pd.read_csv(forwards),
        rates=pd.read_csv(rates),
        spot=pd.read_csv(spot),
        cache_dir=tmp_path / "c4",
    )
    date = "2021-07-21"
    quotes = [
        ds.cross_section(date, days=14).quotes[QUOTE_COMPARE_COLS].reset_index(drop=True)
        for ds in (from_path, from_list, from_dir, from_frame)
    ]
    for other in quotes[1:]:
        pd.testing.assert_frame_equal(quotes[0], other, atol=1e-9, rtol=0)


def test_from_directory_excludes_vix(dataset: Dataset, raw_dir: Path) -> None:
    assert classify_table(raw_dir / "VIX_History.csv") == "spot"
    info = dataset.describe()
    names = [Path(path).name.lower() for path in info["options"]["files"]]
    assert "option_price.csv" in names
    assert all("vix" not in name and not name.startswith("vx") for name in names)
    assert info["spot"] is True
    section = dataset.cross_section("2021-07-21", days=14)
    assert section.spot == pytest.approx(4000.0)
    assert section.spot != pytest.approx(17.01)


def test_from_directory_raw_data_subdir(tmp_path: Path) -> None:
    nested = tmp_path / "DATA"
    write_optionmetrics_dir(nested / "RAW_DATA")
    dataset = Dataset.from_directory(nested, cache_dir=tmp_path / "cache")
    info = dataset.describe()
    assert any(Path(path).name == "option_price.csv" for path in info["options"]["files"])
    assert info["forwards"] == "table"
    assert info["rates"] == "series"
    assert info["spot"] is True


def test_describe_fields(dataset: Dataset, tmp_path: Path) -> None:
    info = dataset.describe()
    assert set(info) == DESCRIBE_KEYS
    assert set(info["options"]) == OPTION_DESCRIBE_KEYS
    assert info["options"]["kind"] == "frame"
    assert info["options"]["n_dates"] == 3
    assert info["options"]["first_date"] == pd.Timestamp("2021-07-21")
    assert info["options"]["last_date"] == pd.Timestamp("2021-07-28")
    assert info["options"]["columns"]["strike_price"] == "strike"
    assert info["date_format"] == "%Y-%m-%d"
    assert info["strike_scale"] == 1000.0
    assert info["forwards"] == "table"
    assert info["rates"] == "series"
    assert info["spot"] is True
    assert info["quote_filter"]["min_volume"] == 1.0
    assert info["cache_dir"] == str(tmp_path / "cache")
    assert isinstance(dataset._source, FrameSource)


def test_available_nearest_and_option_rows(dataset: Dataset) -> None:
    dates = dataset.available_dates()
    assert dates == [
        pd.Timestamp("2021-07-21"),
        pd.Timestamp("2021-07-22"),
        pd.Timestamp("2021-07-28"),
    ]
    assert dataset.nearest_date("2021-07-21") == dates[0]
    assert dataset.nearest_date("2021-07-23") == dates[1]
    assert dataset.nearest_date("2021-07-23", direction="forward") == dates[2]
    with pytest.raises(NoSuchDateError):
        dataset.nearest_date("2021-07-20")
    rows = dataset.option_rows("2021-07-21")
    assert list(rows.columns) == CANONICAL_ROW_COLUMNS
    assert rows["date"].eq(pd.Timestamp("2021-07-21")).all()
    assert rows["strike"].median() < 10_000
    assert set(rows["am_settlement"].unique()) <= {0, 1}


def test_strike_scale_detected_once(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset.option_rows("2021-07-22")
    assert dataset.describe()["strike_scale"] == 1000.0
    calls = {"n": 0}
    real = detect_or_raise

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("crashfrontier.data.dataset.detect_strike_scale", wrapped)
    dataset.option_rows("2021-07-28")
    dataset.option_rows("2021-07-21")
    assert calls["n"] == 0


def detect_or_raise(*args, **kwargs):
    raise AssertionError("strike scale should have been cached")


def test_expiries_keeps_am_pm_separate(dataset: Dataset) -> None:
    table = dataset.expiries("2021-07-21")
    assert list(table.columns) == EXPIRY_COLUMNS
    assert set(table["days"]) == {7, 14, 30}
    thirty = table.loc[table["days"].eq(30)]
    assert set(thirty["am_settlement"]) == {0, 1}
    assert set(thirty["root"]) == {"SPX", "SPXW"}
    assert len(thirty) == 2
    fourteen = table.loc[table["days"].eq(14)].iloc[0]
    assert int(fourteen["am_settlement"]) == 0
    assert fourteen["root"] == "SPXW"
    assert fourteen["forward_source"] == "table"
    assert fourteen["n_put_strikes"] >= 2


def test_cross_section_days_and_target(dataset: Dataset) -> None:
    by_days = dataset.cross_section("2021-07-21", days=14)
    by_expiry = dataset.cross_section("2021-07-21", "2021-08-04")
    by_target = dataset.cross_section("2021-07-21", target_days=14, window=(7, 21))
    assert by_days.days == by_expiry.days == by_target.days == 14
    assert by_days.expiry == pd.Timestamp("2021-08-04")
    assert by_days.am_settlement == 0
    assert by_days.root == "SPXW"
    assert by_days.forward_source == "table"
    assert by_days.rate_source == "series"
    pd.testing.assert_frame_equal(by_days.quotes, by_expiry.quotes)
    pd.testing.assert_frame_equal(by_days.quotes, by_target.quotes)
    assert not bool(by_days.summary()["ambiguous_settlement"])


def test_cross_section_requires_selector(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="expiry, days, or target_days"):
        dataset.cross_section("2021-07-21")


def test_cross_section_missing_expiry(dataset: Dataset) -> None:
    with pytest.raises(NoEligibleExpiryError, match="2021-07-21"):
        dataset.cross_section("2021-07-21", days=99)


def test_cross_sections_does_not_mix_settlement(dataset: Dataset) -> None:
    sections = dataset.cross_sections("2021-07-21")
    assert len(sections) == 4
    keys = {(cs.expiry, cs.am_settlement, cs.root) for cs in sections}
    assert len(keys) == 4
    thirty = [cs for cs in sections if cs.days == 30]
    assert {cs.am_settlement for cs in thirty} == {0, 1}
    assert {cs.root for cs in thirty} == {"SPX", "SPXW"}
    for section in thirty:
        assert section.root != "MIXED"
        assert section.summary()["ambiguous_settlement"] is False


def test_spec_roundtrip(dataset: Dataset) -> None:
    spec = dataset.spec()
    json.dumps(spec)
    assert spec["strike_scale"] == 1000.0
    assert spec["forwards"]
    restored = Dataset.from_spec(spec)
    left = dataset.cross_section("2021-07-21", days=14)
    right = restored.cross_section("2021-07-21", days=14)
    pd.testing.assert_frame_equal(
        left.quotes[QUOTE_COMPARE_COLS].reset_index(drop=True),
        right.quotes[QUOTE_COMPARE_COLS].reset_index(drop=True),
        atol=1e-9,
        rtol=0,
    )
    assert restored.describe()["strike_scale"] == 1000.0


def test_spec_rejects_frames(raw_dir: Path, tmp_path: Path) -> None:
    bundle = write_optionmetrics_dir(tmp_path / "frames", name="frames")
    assert isinstance(bundle, FrameBundle)
    dataset = Dataset.from_frames(
        bundle.options,
        forwards=bundle.forwards,
        rates=bundle.rates,
        spot=bundle.spot,
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="in-memory"):
        dataset.spec()


def test_variant_cross_section_matches_standard(
    dataset: Dataset, variant_dir: Path | FrameBundle, tmp_path: Path
) -> None:
    standard = dataset.cross_section("2021-07-21", days=14)
    variant = _open_variant(variant_dir, tmp_path).cross_section("2021-07-21", days=14)
    left = standard.quotes[QUOTE_COMPARE_COLS].to_numpy(dtype=float)
    right = variant.quotes[QUOTE_COMPARE_COLS].to_numpy(dtype=float)
    assert left.shape == right.shape
    assert np.max(np.abs(left - right)) < 1e-9
    assert variant.days == 14
    assert variant.am_settlement == 0
    assert variant.root != "MIXED"


def test_user_forward_and_rate(dataset: Dataset) -> None:
    section = dataset.cross_section("2021-07-21", days=14, forward=4010.0, rate=0.01)
    assert section.forward == pytest.approx(4010.0)
    assert section.forward_source == "user"
    assert section.rate == pytest.approx(0.01)
    assert section.rate_source == "user"


def test_from_directory_duplicate_forwards(raw_dir: Path, tmp_path: Path) -> None:
    duplicate = raw_dir / "forward_copy.csv"
    duplicate.write_bytes((raw_dir / "forward.csv").read_bytes())
    with pytest.raises(SchemaError, match="multiple forwards"):
        Dataset.from_directory(raw_dir, cache_dir=tmp_path / "cache")
