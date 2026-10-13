from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from crashfrontier.data.cross_section import QUOTE_COLUMNS, CrossSection, build_cross_section
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.schema import ColumnMap, canonicalize
from tests.synthetic import QUOTE_COLUMNS as SYNTHETIC_QUOTE_COLUMNS
from tests.synthetic import make_option_rows, standard_market

FILTER_REPORT_KEYS = [
    "cp_flag",
    "positive_strike",
    "nonnegative_bid",
    "positive_ask",
    "uncrossed",
    "min_volume",
    "min_open_interest",
    "days_window",
    "finite_normalized",
    "min_width",
    "one_quote_per_strike",
    "volume_unavailable",
    "kept",
]
ROW_REPORT_KEYS = [
    "cp_flag",
    "positive_strike",
    "nonnegative_bid",
    "positive_ask",
    "uncrossed",
    "min_volume",
    "min_open_interest",
    "days_window",
    "volume_unavailable",
    "kept",
]
DATE = pd.Timestamp("2021-07-21")
EXPIRY = pd.Timestamp("2021-08-04")
FORWARD = 100.0
RATE = 0.0


def _canonical_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for index, row in enumerate(rows):
        record: dict[str, object] = {
            "date": DATE,
            "expiry": EXPIRY,
            "cp_flag": "P",
            "strike": 90.0,
            "bid": 5.0,
            "ask": 5.2,
            "volume": 10.0,
            "open_interest": 10.0,
            "impl_volatility": 0.18,
            "quote_id": f"q{index}",
            "am_settlement": 0,
            "symbol": "SPXW 210804P90000",
            "root": "SPXW",
            "forward_price": float("nan"),
            "expiry_indicator": "w",
            "days": 14,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _section(rows: pd.DataFrame, quote_filter: QuoteFilter | None = None, **kwargs) -> CrossSection:
    params = {
        "date": DATE,
        "expiry": EXPIRY,
        "am_settlement": 0,
        "forward": FORWARD,
        "forward_source": "user",
        "rate": RATE,
        "rate_source": "user",
        "spot": float("nan"),
        "quote_filter": QuoteFilter.permissive() if quote_filter is None else quote_filter,
    }
    params.update(kwargs)
    return build_cross_section(rows, **params)


def test_quote_columns_match_synthetic_schema() -> None:
    assert QUOTE_COLUMNS == SYNTHETIC_QUOTE_COLUMNS


def test_default_and_permissive_filters() -> None:
    assert QuoteFilter.default() == QuoteFilter()
    assert QuoteFilter.permissive() == QuoteFilter(min_volume=0.0)
    assert QuoteFilter.permissive().min_volume == 0.0
    assert QuoteFilter.default().min_volume == 1.0


def test_apply_rows_counts_and_does_not_mutate() -> None:
    rows = _canonical_frame(
        [
            {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
            {"cp_flag": "P", "strike": 90.0, "bid": 4.0, "ask": 4.2, "volume": 0.0},
            {"cp_flag": "C", "strike": -1.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
            {"cp_flag": "C", "strike": 100.0, "bid": 2.0, "ask": 1.0, "volume": 10.0},
            {"cp_flag": "X", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0},
            {"cp_flag": "C", "strike": 100.0, "bid": -0.5, "ask": 1.2, "volume": 10.0},
            {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 0.0, "volume": 10.0},
            {"cp_flag": "C", "strike": 100.0, "bid": 1.0, "ask": 1.2, "volume": 10.0, "days": 3},
        ]
    )
    original = rows.copy()
    kept, report = QuoteFilter.default().apply_rows(rows)
    pd.testing.assert_frame_equal(rows, original)
    assert list(report) == ROW_REPORT_KEYS
    assert report["cp_flag"] == 1
    assert report["positive_strike"] == 1
    assert report["nonnegative_bid"] == 1
    assert report["positive_ask"] == 1
    assert report["uncrossed"] == 1
    assert report["min_volume"] == 1
    assert report["min_open_interest"] == 0
    assert report["days_window"] == 1
    assert report["volume_unavailable"] == 0
    assert report["kept"] == 1
    assert len(kept) == 1
    assert kept["quote_id"].tolist() == ["q0"]


def test_volume_all_missing_skips_min_volume() -> None:
    rows = _canonical_frame(
        [
            {"volume": float("nan"), "strike": 90.0},
            {"volume": float("nan"), "strike": 95.0},
        ]
    )
    kept, report = QuoteFilter.default().apply_rows(rows)
    assert report["volume_unavailable"] == 1
    assert report["min_volume"] == 0
    assert report["kept"] == 2
    assert len(kept) == 2

    no_volume = rows.drop(columns=["volume"])
    kept_missing, report_missing = QuoteFilter.default().apply_rows(no_volume)
    assert report_missing["volume_unavailable"] == 1
    assert report_missing["min_volume"] == 0
    assert report_missing["kept"] == 2
    assert len(kept_missing) == 2


def test_permissive_keeps_zero_volume() -> None:
    rows = _canonical_frame([{"volume": 0.0}, {"volume": 10.0}])
    kept_default, report_default = QuoteFilter.default().apply_rows(rows)
    kept_perm, report_perm = QuoteFilter.permissive().apply_rows(rows)
    assert report_default["min_volume"] == 1
    assert report_default["kept"] == 1
    assert report_perm["min_volume"] == 0
    assert report_perm["kept"] == 2
    assert len(kept_default) == 1
    assert len(kept_perm) == 2


def test_min_open_interest_and_cp_flags() -> None:
    rows = _canonical_frame(
        [
            {"cp_flag": "C", "open_interest": 0.0},
            {"cp_flag": "P", "open_interest": 8.0},
        ]
    )
    kept, report = QuoteFilter(min_volume=0.0, min_open_interest=1.0, cp_flags=("P",)).apply_rows(rows)
    assert report["cp_flag"] == 1
    assert report["min_open_interest"] == 0
    assert report["kept"] == 1
    assert kept["cp_flag"].tolist() == ["P"]

    kept_oi, report_oi = QuoteFilter(min_volume=0.0, min_open_interest=1.0).apply_rows(rows)
    assert report_oi["min_open_interest"] == 1
    assert kept_oi["cp_flag"].tolist() == ["P"]


def test_one_quote_per_strike_keeps_narrowest() -> None:
    quotes = pd.DataFrame(
        {
            "quote_id": ["wide", "narrow", "other", "tie_a", "tie_b"],
            "symbol": [""] * 5,
            "root": [""] * 5,
            "cp_flag": ["P", "P", "C", "P", "P"],
            "strike": [90.0, 90.0, 90.0, 80.0, 80.0],
            "bid": [1.0, 1.1, 0.5, 2.0, 2.0],
            "ask": [1.4, 1.2, 0.6, 2.3, 2.3],
            "volume": [10.0] * 5,
            "open_interest": [10.0] * 5,
            "impl_volatility": [np.nan] * 5,
            "x": [0.9, 0.9, 0.9, 0.8, 0.8],
            "log_moneyness": np.log([0.9, 0.9, 0.9, 0.8, 0.8]),
            "norm_bid": [0.20, 0.21, 0.05, 0.30, 0.30],
            "norm_ask": [0.28, 0.23, 0.06, 0.35, 0.35],
            "norm_l": [0.20, 0.21, 0.05, 0.30, 0.30],
            "norm_u": [0.28, 0.23, 0.06, 0.35, 0.35],
            "role": ["otm_put", "otm_put", "itm", "otm_put", "otm_put"],
        }
    )
    kept, report = QuoteFilter(one_quote_per_strike=True).apply_normalized(quotes)
    assert report["one_quote_per_strike"] == 2
    assert report["kept"] == 3
    assert set(kept["quote_id"]) == {"narrow", "other", "tie_a"}
    pair = kept.loc[kept["strike"].eq(90.0) & kept["cp_flag"].eq("P")]
    assert pair["quote_id"].tolist() == ["narrow"]


def test_finite_normalized_and_min_width() -> None:
    quotes = pd.DataFrame(
        {
            "quote_id": ["ok", "inf_x", "crossed", "thin"],
            "symbol": [""] * 4,
            "root": [""] * 4,
            "cp_flag": ["P"] * 4,
            "strike": [90.0, 90.0, 90.0, 90.0],
            "bid": [1.0, 1.0, 1.0, 1.0],
            "ask": [1.2, 1.2, 1.2, 1.01],
            "volume": [10.0] * 4,
            "open_interest": [10.0] * 4,
            "impl_volatility": [np.nan] * 4,
            "x": [0.9, np.inf, 0.9, 0.9],
            "log_moneyness": [math.log(0.9), np.nan, math.log(0.9), math.log(0.9)],
            "norm_bid": [0.10, 0.10, 0.20, 0.10],
            "norm_ask": [0.12, 0.12, 0.11, 0.101],
            "norm_l": [0.10, 0.10, 0.20, 0.10],
            "norm_u": [0.12, 0.12, 0.11, 0.101],
            "role": ["otm_put"] * 4,
        }
    )
    kept, report = QuoteFilter(min_width=0.015).apply_normalized(quotes)
    assert report["finite_normalized"] == 2
    assert report["min_width"] == 1
    assert report["kept"] == 1
    assert kept["quote_id"].tolist() == ["ok"]


def test_norm_l_norm_u_formulas() -> None:
    rows = _canonical_frame(
        [
            {"cp_flag": "C", "strike": 110.0, "bid": 2.0, "ask": 2.5, "quote_id": "c"},
            {"cp_flag": "P", "strike": 90.0, "bid": 4.0, "ask": 4.4, "quote_id": "p"},
        ]
    )
    section = _section(rows, rate=0.05)
    discount = math.exp(-0.05 * 14.0 / 365.0)
    scale = discount * FORWARD
    calls = section.calls.iloc[0]
    puts = section.puts.iloc[0]
    assert calls["x"] == pytest.approx(110.0 / FORWARD)
    assert puts["x"] == pytest.approx(90.0 / FORWARD)
    assert calls["log_moneyness"] == pytest.approx(math.log(1.1))
    assert puts["log_moneyness"] == pytest.approx(math.log(0.9))
    assert calls["norm_bid"] == pytest.approx(2.0 / scale)
    assert calls["norm_ask"] == pytest.approx(2.5 / scale)
    assert calls["norm_l"] == pytest.approx(calls["norm_bid"])
    assert calls["norm_u"] == pytest.approx(calls["norm_ask"])
    assert puts["norm_l"] == pytest.approx(puts["norm_bid"] + 1.0 - puts["x"])
    assert puts["norm_u"] == pytest.approx(puts["norm_ask"] + 1.0 - puts["x"])
    assert calls["role"] == "otm_call"
    assert puts["role"] == "otm_put"


def test_put_call_parity_matches_when_bid_equals_ask() -> None:
    strike = 90.0
    put_price = 5.0
    call_price = put_price + 1.0 * (FORWARD - strike)
    rows = _canonical_frame(
        [
            {"cp_flag": "C", "strike": strike, "bid": call_price, "ask": call_price, "quote_id": "c"},
            {"cp_flag": "P", "strike": strike, "bid": put_price, "ask": put_price, "quote_id": "p"},
        ]
    )
    section = _section(rows)
    call_row = section.calls.iloc[0]
    put_row = section.puts.iloc[0]
    assert call_row["norm_l"] == pytest.approx(put_row["norm_l"])
    assert call_row["norm_u"] == pytest.approx(put_row["norm_u"])
    assert call_row["norm_l"] == pytest.approx(call_row["norm_u"])
    assert put_row["norm_l"] == pytest.approx(put_row["norm_u"])


def test_role_atm_and_itm() -> None:
    rows = _canonical_frame(
        [
            {"cp_flag": "P", "strike": 100.0, "quote_id": "atm_p"},
            {"cp_flag": "C", "strike": 100.0, "quote_id": "atm_c"},
            {"cp_flag": "P", "strike": 110.0, "quote_id": "itm_p"},
            {"cp_flag": "C", "strike": 90.0, "quote_id": "itm_c"},
        ]
    )
    section = _section(rows)
    by_id = section.quotes.set_index("quote_id")["role"]
    assert by_id["atm_p"] == "otm_put"
    assert by_id["atm_c"] == "otm_call"
    assert by_id["itm_p"] == "itm"
    assert by_id["itm_c"] == "itm"


def test_quotes_column_order_and_sort() -> None:
    rows = _canonical_frame(
        [
            {"cp_flag": "P", "strike": 95.0, "quote_id": "p95"},
            {"cp_flag": "C", "strike": 90.0, "quote_id": "c90"},
            {"cp_flag": "P", "strike": 90.0, "quote_id": "p90"},
        ]
    )
    section = _section(rows)
    assert list(section.quotes.columns) == list(QUOTE_COLUMNS)
    assert section.quotes["strike"].tolist() == [90.0, 90.0, 95.0]
    assert section.quotes["cp_flag"].tolist() == ["C", "P", "P"]
    assert section.quotes["quote_id"].tolist() == ["c90", "p90", "p95"]


def test_mixed_root_summary_and_repr() -> None:
    rows = _canonical_frame(
        [
            {"root": "SPX", "symbol": "SPX 210804P90000", "cp_flag": "P"},
            {"root": "SPXW", "symbol": "SPXW 210804C110000", "cp_flag": "C", "strike": 110.0},
        ]
    )
    section = _section(rows, spot=99.5)
    assert section.root == "MIXED"
    assert section.days == 14
    assert section.tau == pytest.approx(14.0 / 365.0)
    assert section.discount == pytest.approx(1.0)
    assert section.forward_source == "user"
    assert section.rate_source == "user"
    assert list(section.filter_report) == FILTER_REPORT_KEYS
    assert section.filter_report["kept"] == 2
    summary = section.summary()
    assert summary["n_quotes"] == 2
    assert summary["n_puts"] == 1
    assert summary["n_calls"] == 1
    assert summary["n_put_strikes"] == 1
    assert summary["n_call_strikes"] == 1
    assert summary["root"] == "MIXED"
    assert summary["spot"] == pytest.approx(99.5)
    text = repr(section)
    assert "date=2021-07-21" in text
    assert "expiry=2021-08-04" in text
    assert "days=14" in text
    assert "root=MIXED" in text
    assert "F=100.0" in text
    assert "n_puts=1" in text
    assert "n_calls=1" in text


def test_from_quotes_matches_build_cross_section() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "exdate": ["2021-08-04", "2021-08-04"],
            "cp_flag": ["C", "P"],
            "strike": [110.0, 90.0],
            "bid": [1.5, 4.0],
            "ask": [1.7, 4.2],
            "volume": [10, 8],
            "open_interest": [3, 5],
            "optionid": [11, 12],
            "am_settlement": [0, 0],
            "symbol": ["SPXW 210804C110000", "SPXW 210804P90000"],
        }
    )
    quote_filter = QuoteFilter.permissive()
    canonical = canonicalize(
        raw,
        ColumnMap.detect(raw.columns),
        strike_scale=1.0,
        dayfirst=None,
        cp_values=None,
    )
    built = build_cross_section(
        canonical,
        date=DATE,
        expiry=EXPIRY,
        am_settlement=0,
        forward=FORWARD,
        forward_source="user",
        rate=RATE,
        rate_source="user",
        spot=float("nan"),
        quote_filter=quote_filter,
    )
    from_table = CrossSection.from_quotes(
        raw,
        date="2021-07-21",
        expiry="2021-08-04",
        forward=FORWARD,
        rate=RATE,
        am_settlement=0,
        quote_filter=quote_filter,
    )
    pd.testing.assert_frame_equal(from_table.quotes, built.quotes)
    assert from_table.days == built.days == 14
    assert from_table.root == built.root == "SPXW"
    assert from_table.forward_source == "user"
    assert from_table.rate_source == "user"
    assert from_table.filter_report == built.filter_report


def test_from_quotes_fills_missing_date_expiry_and_scale() -> None:
    raw = pd.DataFrame(
        {
            "cp_flag": ["P"],
            "strike_price": [90000.0],
            "best_bid": [5.0],
            "best_offer": [5.1],
            "volume": [4],
        }
    )
    section = CrossSection.from_quotes(
        raw,
        date="2021-07-21",
        expiry="2021-08-04",
        forward=FORWARD,
        rate=RATE,
        strike_scale=1000.0,
        quote_filter=QuoteFilter.permissive(),
    )
    assert section.quotes["strike"].tolist() == [90.0]
    assert section.date == DATE
    assert section.expiry == EXPIRY
    assert section.am_settlement == 0
    assert section.filter_report["volume_unavailable"] == 0


def test_from_quotes_applies_dayfirst_to_date_arguments() -> None:
    raw = pd.DataFrame(
        {
            "cp_flag": ["P"],
            "strike": [90.0],
            "bid": [5.0],
            "ask": [5.2],
            "volume": [10],
        }
    )
    section = CrossSection.from_quotes(
        raw,
        date="05/07/2021",
        expiry="19/07/2021",
        forward=100.0,
        rate=0.053,
        dayfirst=True,
        quote_filter=QuoteFilter.permissive(),
    )
    assert section.date == pd.Timestamp("2021-07-05")
    assert section.expiry == pd.Timestamp("2021-07-19")
    assert section.days == 14
    assert section.discount == pytest.approx(np.exp(-0.053 * 14 / 365))
    assert len(section.quotes) == 1


def test_from_quotes_skips_volume_when_column_absent() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2021-07-21"],
            "expiry": ["2021-08-04"],
            "cp_flag": ["P"],
            "strike": [90.0],
            "bid": [5.0],
            "ask": [5.2],
        }
    )
    section = CrossSection.from_quotes(
        raw,
        date=DATE,
        expiry=EXPIRY,
        forward=FORWARD,
        quote_filter=QuoteFilter.default(),
    )
    assert section.filter_report["volume_unavailable"] == 1
    assert section.filter_report["min_volume"] == 0
    assert section.filter_report["kept"] == 1


def test_one_quote_per_strike_through_build_cross_section() -> None:
    rows = _canonical_frame(
        [
            {
                "cp_flag": "P",
                "strike": 90.0,
                "bid": 4.0,
                "ask": 5.0,
                "quote_id": "wide",
            },
            {
                "cp_flag": "P",
                "strike": 90.0,
                "bid": 4.4,
                "ask": 4.6,
                "quote_id": "narrow",
            },
            {
                "cp_flag": "C",
                "strike": 90.0,
                "bid": 11.0,
                "ask": 11.5,
                "quote_id": "call",
            },
        ]
    )
    section = _section(rows, QuoteFilter(min_volume=0.0, one_quote_per_strike=True))
    assert section.filter_report["one_quote_per_strike"] == 1
    puts = section.puts
    assert len(puts) == 1
    assert puts["quote_id"].tolist() == ["narrow"]
    assert len(section.calls) == 1


def test_synthetic_optionmetrics_from_quotes_matches_build() -> None:
    dates, expiries_by_date, forwards, rates = standard_market()
    raw = make_option_rows(dates, expiries_by_date, forwards, rates)
    date = dates[0]
    expiry = date + pd.Timedelta(days=14)
    forward = forwards[(date, expiry, 0)]
    rate = rates[date]
    mask = (
        raw["date"].eq(date.strftime("%Y-%m-%d"))
        & raw["exdate"].eq(expiry.strftime("%Y-%m-%d"))
        & raw["am_settlement"].eq(0)
    )
    raw_slice = raw.loc[mask].reset_index(drop=True)
    quote_filter = QuoteFilter.default()
    canonical = canonicalize(
        raw_slice,
        ColumnMap.detect(raw_slice.columns),
        strike_scale=1000.0,
        dayfirst=None,
        cp_values=None,
    )
    built = build_cross_section(
        canonical,
        date=date,
        expiry=expiry,
        am_settlement=0,
        forward=forward,
        forward_source="user",
        rate=rate,
        rate_source="user",
        spot=float("nan"),
        quote_filter=quote_filter,
    )
    from_table = CrossSection.from_quotes(
        raw_slice,
        date=date,
        expiry=expiry,
        forward=forward,
        rate=rate,
        am_settlement=0,
        strike_scale=1000.0,
        quote_filter=quote_filter,
    )
    pd.testing.assert_frame_equal(from_table.quotes, built.quotes)
    assert (built.quotes["volume"] >= 1.0).all()
    assert built.root == "SPXW"
    assert built.days == 14
    assert built.filter_report["min_volume"] > 0
    assert list(built.quotes.columns) == list(QUOTE_COLUMNS)
    assert built.filter_report["kept"] == len(built.quotes)


def test_dataset_cross_section_matches_from_quotes(dataset) -> None:
    section = dataset.cross_section("2021-07-21", days=14)
    dates, expiries_by_date, forwards, rates = standard_market()
    raw = make_option_rows(dates, expiries_by_date, forwards, rates)
    date = dates[0]
    expiry = date + pd.Timedelta(days=14)
    mask = (
        raw["date"].eq(date.strftime("%Y-%m-%d"))
        & raw["exdate"].eq(expiry.strftime("%Y-%m-%d"))
        & raw["am_settlement"].eq(0)
    )
    from_table = CrossSection.from_quotes(
        raw.loc[mask].reset_index(drop=True),
        date=date,
        expiry=expiry,
        forward=section.forward,
        rate=section.rate,
        am_settlement=0,
        strike_scale=1000.0,
        quote_filter=QuoteFilter.default(),
    )
    pd.testing.assert_frame_equal(from_table.quotes, section.quotes)
    assert from_table.days == section.days == 14
    assert from_table.root == section.root == "SPXW"
    assert section.forward_source == "table"
    assert from_table.forward_source == "user"


def test_ambiguous_settlement_does_not_mix_am_pm(dataset, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="crashfrontier"):
        section = dataset.cross_section("2021-07-21", days=30)
    assert section.days == 30
    assert section.root in {"SPX", "SPXW"}
    assert section.root != "MIXED"
    assert section.am_settlement in {0, 1}
    assert section.summary()["ambiguous_settlement"] is True
    assert "ambiguous AM/PM settlement" in caplog.text
    specified = dataset.cross_section("2021-07-21", days=30, am_settlement=0)
    assert specified.am_settlement == 0
    assert specified.root == "SPXW"
    assert specified.summary()["ambiguous_settlement"] is False
    other = dataset.cross_section("2021-07-21", days=30, am_settlement=1)
    assert other.am_settlement == 1
    assert other.root == "SPX"
    assert specified.quotes["quote_id"].tolist() != other.quotes["quote_id"].tolist()
    rows = dataset.option_rows("2021-07-21")
    group = rows.loc[
        rows["expiry"].eq(section.expiry) & rows["am_settlement"].eq(section.am_settlement)
    ]
    assert set(section.quotes["quote_id"]).issubset(set(group["quote_id"]))
    mixed = rows.loc[rows["expiry"].eq(section.expiry)]
    assert mixed["am_settlement"].nunique() == 2
    assert section.quotes["quote_id"].nunique() < mixed["quote_id"].nunique()
