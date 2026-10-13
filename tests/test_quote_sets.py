from __future__ import annotations

import math

import pandas as pd
import pytest

from crashfrontier.data.cross_section import QUOTE_COLUMNS
from crashfrontier.data.quote_sets import (
    full_otm,
    full_panel,
    local_eight,
    nested_quote_sets,
    otm_call_universe,
    otm_put_universe,
    put_pair,
    put_universe,
    select_quote_set,
)

THRESHOLD = 0.90


def _quotes(rows: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for index, row in enumerate(rows):
        x = float(row.get("x", row.get("strike", 0.9)))
        cp_flag = str(row.get("cp_flag", "P"))
        record: dict[str, object] = {
            "quote_id": f"q{index}",
            "symbol": "",
            "root": "SPXW",
            "cp_flag": cp_flag,
            "strike": x,
            "bid": 0.10,
            "ask": 0.12,
            "volume": 10.0,
            "open_interest": 10.0,
            "impl_volatility": 0.18,
            "x": x,
            "log_moneyness": math.log(x) if x > 0.0 else float("nan"),
            "norm_bid": 0.10,
            "norm_ask": 0.12,
            "norm_l": 0.10,
            "norm_u": 0.12,
            "role": _role(cp_flag, x),
        }
        record.update(row)
        if "strike" not in row:
            record["strike"] = record["x"]
        if "log_moneyness" not in row and float(record["x"]) > 0.0:
            record["log_moneyness"] = math.log(float(record["x"]))
        records.append(record)
    return pd.DataFrame.from_records(records).loc[:, list(QUOTE_COLUMNS)]


def _role(cp_flag: str, x: float) -> str:
    if cp_flag == "P" and x <= 1.0:
        return "otm_put"
    if cp_flag == "C" and x >= 1.0:
        return "otm_call"
    return "itm"


def _ids(frame: pd.DataFrame) -> set[str]:
    return set(frame["quote_id"].tolist())


def _strikes(frame: pd.DataFrame) -> list[float]:
    return sorted(frame["x"].unique().tolist())


def _panel() -> pd.DataFrame:
    put_x = [
        0.70,
        0.72,
        0.76,
        0.80,
        0.84,
        0.88,
        0.90,
        0.95,
        0.98,
        1.00,
        1.04,
        1.10,
    ]
    rows: list[dict[str, object]] = []
    for x in put_x:
        rows.append({"cp_flag": "P", "x": x, "quote_id": f"p{x:.2f}"})
    rows.append({"cp_flag": "P", "x": 0.88, "quote_id": "p0.88b"})
    for x in (0.95, 1.00, 1.10):
        rows.append({"cp_flag": "C", "x": x, "quote_id": f"c{x:.2f}"})
    return _quotes(rows)


def test_put_universe_keeps_puts_inside_log_moneyness() -> None:
    quotes = _panel()
    kept = put_universe(quotes)
    assert (kept["cp_flag"] == "P").all()
    assert kept["log_moneyness"].min() >= -0.35
    assert kept["log_moneyness"].max() <= 0.05
    assert "p0.70" not in _ids(kept)
    assert "p1.10" not in _ids(kept)
    assert "c1.00" not in _ids(kept)
    assert "p1.04" in _ids(kept)
    assert "p0.88" in _ids(kept)
    assert "p0.88b" in _ids(kept)


def test_otm_put_and_call_universes() -> None:
    quotes = _panel()
    puts = otm_put_universe(quotes)
    calls = otm_call_universe(quotes)
    assert (puts["cp_flag"] == "P").all()
    assert (puts["x"] <= 1.0).all()
    assert puts["log_moneyness"].min() >= -0.35
    assert "p1.04" not in _ids(puts)
    assert "p0.70" not in _ids(puts)
    assert (calls["cp_flag"] == "C").all()
    assert (calls["x"] >= 1.0).all()
    assert _ids(calls) == {"c1.00", "c1.10"}
    assert "c0.95" not in _ids(calls)


def test_full_otm_is_union_and_full_panel_keeps_calls_and_puts() -> None:
    quotes = _panel()
    otm = full_otm(quotes)
    puts = otm_put_universe(quotes)
    calls = otm_call_universe(quotes)
    assert _ids(otm) == _ids(puts) | _ids(calls)
    panel = full_panel(quotes)
    assert _ids(panel) == _ids(quotes)
    assert list(panel.columns) == list(QUOTE_COLUMNS)


def test_put_pair_rule1_largest_below_and_smallest_above() -> None:
    quotes = _quotes(
        [
            {"x": 0.70, "quote_id": "far_below"},
            {"x": 0.80, "quote_id": "mid_below"},
            {"x": 0.85, "quote_id": "near_below"},
            {"x": 0.93, "quote_id": "near_above"},
            {"x": 1.00, "quote_id": "far_above"},
        ]
    )
    pair = put_pair(quotes, THRESHOLD)
    assert _strikes(pair) == [0.85, 0.93]
    assert _ids(pair) == {"near_below", "near_above"}


def test_put_pair_rule2_keeps_bracketing_strikes_not_two_nearest() -> None:
    quotes = _quotes(
        [
            {"x": 0.70, "quote_id": "closer_below"},
            {"x": 0.88, "quote_id": "below"},
            {"x": 1.20, "quote_id": "above"},
        ]
    )
    pair = put_pair(quotes, THRESHOLD)
    assert _strikes(pair) == [0.88, 1.20]
    assert "closer_below" not in _ids(pair)


def test_put_pair_rule3_threshold_strike_plus_nearest_other() -> None:
    quotes = _quotes(
        [
            {"x": 0.80, "quote_id": "near"},
            {"x": 0.90, "quote_id": "at_k"},
            {"x": 1.10, "quote_id": "far"},
        ]
    )
    pair = put_pair(quotes, THRESHOLD)
    assert _strikes(pair) == [0.80, 0.90]
    assert _ids(pair) == {"near", "at_k"}


def test_physical_put_pair_uses_strictly_bracketing_strikes() -> None:
    sets = nested_quote_sets(_panel(), THRESHOLD, design="physical")
    assert _strikes(sets["put_pair"]) == [0.88, 0.95]
    assert 0.90 not in _strikes(sets["put_pair"])
    assert _ids(sets["put_pair"]) <= _ids(sets["local_eight"])


def test_full_otm_selection_does_not_require_a_bracketing_pair() -> None:
    quotes = _quotes(
        [
            {"x": 0.70, "quote_id": "far"},
            {"x": 0.80, "quote_id": "mid"},
            {"x": 0.85, "quote_id": "near"},
            {"x": 1.05, "cp_flag": "C", "quote_id": "call"},
        ]
    )
    selected = select_quote_set(quotes, THRESHOLD, "full_otm", "physical")
    assert _ids(selected) == {"mid", "near", "call"}


def test_put_pair_rule4_two_nearest_when_unbracketed() -> None:
    quotes = _quotes(
        [
            {"x": 0.70, "quote_id": "far"},
            {"x": 0.80, "quote_id": "mid"},
            {"x": 0.85, "quote_id": "near"},
        ]
    )
    pair = put_pair(quotes, THRESHOLD)
    assert _strikes(pair) == [0.80, 0.85]
    assert _ids(pair) == {"mid", "near"}

    above_only = _quotes(
        [
            {"x": 0.95, "quote_id": "near_above"},
            {"x": 1.05, "quote_id": "mid_above"},
            {"x": 1.20, "quote_id": "far_above"},
        ]
    )
    pair_above = put_pair(above_only, THRESHOLD)
    assert _strikes(pair_above) == [0.95, 1.05]


def test_put_pair_rule5_keeps_every_row_at_selected_strikes() -> None:
    quotes = _quotes(
        [
            {"x": 0.85, "quote_id": "lo_a"},
            {"x": 0.85, "quote_id": "lo_b"},
            {"x": 0.95, "quote_id": "hi"},
            {"x": 0.70, "quote_id": "other"},
        ]
    )
    pair = put_pair(quotes, THRESHOLD)
    assert _strikes(pair) == [0.85, 0.95]
    assert pair["quote_id"].tolist() == ["lo_a", "lo_b", "hi"]
    assert len(pair) == 3


def test_local_eight_has_eight_strikes_and_contains_pair() -> None:
    quotes = _panel()
    puts = put_universe(quotes)
    pair = put_pair(puts, THRESHOLD)
    local = local_eight(puts, THRESHOLD)
    assert puts["x"].nunique() > 8
    assert local["x"].nunique() == 8
    assert _ids(pair) <= _ids(local) <= _ids(puts)
    assert set(pair["x"]) <= set(local["x"])
    expected = local.sort_values(["strike", "cp_flag", "quote_id"], kind="mergesort")
    pd.testing.assert_frame_equal(local, expected.reset_index(drop=True))


def test_local_eight_duplicate_rows_do_not_count_as_extra_strikes() -> None:
    quotes = _quotes(
        [{"x": 0.72 + 0.04 * index, "quote_id": f"s{index}"} for index in range(10)]
        + [{"x": 0.88, "quote_id": "dup"}]
    )
    local = local_eight(quotes, THRESHOLD)
    assert local["x"].nunique() == 8
    if 0.88 in set(local["x"]):
        assert local.loc[local["x"].eq(0.88), "quote_id"].tolist() == ["dup", "s4"]


def test_nested_quote_sets_are_nested() -> None:
    quotes = _panel()
    riskneutral = nested_quote_sets(quotes, THRESHOLD, "riskneutral")
    assert list(riskneutral) == ["put_pair", "local_eight", "put_wing"]
    assert _ids(riskneutral["put_pair"]) <= _ids(riskneutral["local_eight"])
    assert _ids(riskneutral["local_eight"]) <= _ids(riskneutral["put_wing"])
    assert riskneutral["local_eight"]["x"].nunique() == 8
    assert riskneutral["put_wing"]["x"].nunique() >= 8
    assert (riskneutral["put_wing"]["cp_flag"] == "P").all()
    assert "p1.04" in _ids(riskneutral["put_wing"])
    assert "p1.10" not in _ids(riskneutral["put_wing"])

    physical = nested_quote_sets(quotes, THRESHOLD, "physical")
    assert list(physical) == ["put_pair", "local_eight", "put_wing", "full_otm"]
    assert _ids(physical["put_pair"]) <= _ids(physical["local_eight"])
    assert _ids(physical["local_eight"]) <= _ids(physical["put_wing"])
    assert _ids(physical["put_wing"]) <= _ids(physical["full_otm"])
    assert "p1.04" not in _ids(physical["put_wing"])
    assert "c1.00" in _ids(physical["full_otm"])
    assert "c0.95" not in _ids(physical["full_otm"])


def test_select_quote_set_returns_named_collection() -> None:
    quotes = _panel()
    nested = nested_quote_sets(quotes, THRESHOLD, "physical")
    pd.testing.assert_frame_equal(
        select_quote_set(quotes, THRESHOLD, "local_eight", "physical"),
        nested["local_eight"],
    )
    panel = select_quote_set(quotes, THRESHOLD, "full_panel", "riskneutral")
    pd.testing.assert_frame_equal(panel, full_panel(quotes))


def test_select_quote_set_unknown_name_lists_available() -> None:
    quotes = _panel()
    with pytest.raises(ValueError, match="unknown quote set 'vertical_pair'") as info:
        select_quote_set(quotes, THRESHOLD, "vertical_pair", "riskneutral")
    message = str(info.value)
    assert "put_pair" in message
    assert "local_eight" in message
    assert "put_wing" in message
    assert "full_panel" in message
    assert "full_otm" not in message

    with pytest.raises(ValueError, match="full_otm"):
        select_quote_set(quotes, THRESHOLD, "full_otm", "riskneutral")
    with pytest.raises(ValueError, match="unknown design"):
        nested_quote_sets(quotes, THRESHOLD, "gamma0")


def test_outputs_sorted_reset_and_keep_all_columns() -> None:
    quotes = _quotes(
        [
            {"x": 0.95, "cp_flag": "P", "quote_id": "p95"},
            {"x": 0.85, "cp_flag": "P", "quote_id": "p85b"},
            {"x": 0.85, "cp_flag": "C", "quote_id": "c85"},
            {"x": 0.85, "cp_flag": "P", "quote_id": "p85a"},
            {"x": 1.05, "cp_flag": "C", "quote_id": "c105"},
        ]
    )
    original = quotes.copy()
    panel = full_panel(quotes)
    pd.testing.assert_frame_equal(quotes, original)
    assert panel["strike"].tolist() == [0.85, 0.85, 0.85, 0.95, 1.05]
    assert panel["cp_flag"].tolist() == ["C", "P", "P", "P", "C"]
    assert panel["quote_id"].tolist() == ["c85", "p85a", "p85b", "p95", "c105"]
    assert list(panel.index) == list(range(len(panel)))
    assert list(panel.columns) == list(quotes.columns)

    pair = put_pair(quotes, THRESHOLD)
    assert pair["quote_id"].tolist() == ["p85a", "p85b", "p95"]
    pd.testing.assert_frame_equal(quotes, original)
