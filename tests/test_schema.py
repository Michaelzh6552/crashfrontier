from __future__ import annotations

import pandas as pd
import pytest

from crashfrontier.data.schema import (
    ALIASES,
    CANONICAL_FIELDS,
    REQUIRED_FIELDS,
    ColumnMap,
    canonicalize,
    normalize_name,
)
from crashfrontier.errors import SchemaError

OPTIONMETRICS_HEADER = (
    "secid,date,symbol,symbol_flag,exdate,last_date,cp_flag,strike_price,"
    "best_bid,best_offer,volume,open_interest,impl_volatility,delta,gamma,vega,"
    "theta,optionid,cfadj,am_settlement,contract_size,ss_flag,forward_price,"
    "expiry_indicator,root,suffix,cusip,ticker,sic,index_flag,exchange_d,class,"
    "issue_type,industry_group,issuer,div_convention,exercise_style,am_set_flag"
)

OUTPUT_COLUMNS = [
    "date",
    "expiry",
    "cp_flag",
    "strike",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "impl_volatility",
    "quote_id",
    "am_settlement",
    "symbol",
    "root",
    "forward_price",
    "expiry_indicator",
    "days",
]


def test_normalize_name_strips_bom_quotes_and_punctuation() -> None:
    assert normalize_name("\ufeffDate") == "date"
    assert normalize_name('"Best Bid"') == "best_bid"
    assert normalize_name("AM-Set.Flag") == "am_set_flag"
    assert normalize_name("  Expiration  ") == "expiration"


def test_detect_standard_optionmetrics_names() -> None:
    columns = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "optionid",
        "am_settlement",
        "symbol",
        "root",
        "forward_price",
        "expiry_indicator",
    ]
    mapping = ColumnMap.detect(columns)
    assert mapping == ColumnMap.optionmetrics()
    assert mapping.usecols() == columns
    assert mapping.rename_map()["exdate"] == "expiry"
    assert mapping.rename_map()["best_offer"] == "ask"


def test_detect_uppercase_names_preserve_original_spelling() -> None:
    mapping = ColumnMap.detect(
        ["DATE", "EXDATE", "CP_FLAG", "STRIKE_PRICE", "BEST_BID", "BEST_OFFER", "VOLUME"]
    )
    assert mapping.date == "DATE"
    assert mapping.expiry == "EXDATE"
    assert mapping.cp_flag == "CP_FLAG"
    assert mapping.strike == "STRIKE_PRICE"
    assert mapping.bid == "BEST_BID"
    assert mapping.ask == "BEST_OFFER"
    assert mapping.volume == "VOLUME"


def test_detect_aliases_bid_ask_expiration() -> None:
    mapping = ColumnMap.detect(["date", "Expiration", "cp_flag", "strike", "Bid", "Ask"])
    assert mapping.expiry == "Expiration"
    assert mapping.bid == "Bid"
    assert mapping.ask == "Ask"
    assert mapping.strike == "strike"


def test_detect_missing_required_includes_aliases_and_columns() -> None:
    columns = ["date", "cp_flag", "strike", "Bid", "Ask"]
    with pytest.raises(SchemaError) as info:
        ColumnMap.detect(columns)
    message = str(info.value)
    assert "expiry" in message
    for alias in ALIASES["expiry"]:
        assert alias in message
    for column in columns:
        assert column in message


def test_detect_optionmetrics_header_from_spec() -> None:
    columns = OPTIONMETRICS_HEADER.split(",")
    assert len(columns) == 38
    mapping = ColumnMap.detect(columns)
    assert mapping.am_settlement == "am_settlement"
    assert mapping.am_settlement != "am_set_flag"
    assert mapping.quote_id == "optionid"
    assert mapping.forward_price == "forward_price"
    assert mapping.expiry == "exdate"
    assert mapping.strike == "strike_price"
    assert mapping.bid == "best_bid"
    assert mapping.ask == "best_offer"
    assert mapping.symbol == "symbol"
    assert mapping.root == "root"


def test_detect_short_aliases_only_when_long_aliases_absent() -> None:
    with_long = ColumnMap.detect(
        ["date", "exdate", "cp_flag", "strike_price", "bid", "ask", "type", "k", "id"]
    )
    assert with_long.cp_flag == "cp_flag"
    assert with_long.strike == "strike_price"
    assert with_long.quote_id == "id"
    short_only = ColumnMap.detect(["date", "expiry", "type", "k", "Bid", "Ask"])
    assert short_only.cp_flag == "type"
    assert short_only.strike == "k"


def test_detect_conflict_treats_later_field_as_missing() -> None:
    columns = ["date", "cp_flag", "strike", "bid", "ask"]
    with pytest.raises(SchemaError, match="missing required field 'expiry'"):
        ColumnMap.detect(columns, overrides={"expiry": "date"})


def test_detect_overrides_win() -> None:
    mapping = ColumnMap.detect(
        ["date", "MyDate", "exdate", "cp_flag", "strike", "bid", "ask"],
        overrides={"date": "MyDate"},
    )
    assert mapping.date == "MyDate"


def test_canonicalize_output_columns_and_types() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "exdate": ["2021-08-04", "2021-08-04"],
            "cp_flag": ["C", "P"],
            "strike_price": [4000000.0, 3900000.0],
            "best_bid": [10.0, 11.5],
            "best_offer": [10.2, 11.7],
            "volume": [2, 0],
            "open_interest": [8, 3],
            "impl_volatility": [0.18, 0.21],
            "optionid": [83164582.0, 83164583.0],
            "am_settlement": [1, 0],
            "symbol": ["SPX 210804C4000000", "SPXW 210804P3900000"],
            "forward_price": [pd.NA, pd.NA],
            "expiry_indicator": ["", "w"],
        }
    )
    out = canonicalize(
        raw,
        ColumnMap.detect(raw.columns),
        strike_scale=1000.0,
        dayfirst=None,
        cp_values=None,
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert str(out["date"].dtype).startswith("datetime64")
    assert str(out["expiry"].dtype).startswith("datetime64")
    assert out["cp_flag"].tolist() == ["C", "P"]
    assert out["strike"].dtype == "float64"
    assert out["strike"].tolist() == [4000.0, 3900.0]
    assert out["bid"].dtype == "float64"
    assert out["ask"].dtype == "float64"
    assert out["quote_id"].tolist() == ["83164582", "83164583"]
    assert out["am_settlement"].tolist() == [1, 0]
    assert int(out["am_settlement"].dtype.kind == "i" or str(out["am_settlement"].dtype).startswith("int"))
    assert out["root"].tolist() == ["SPX", "SPXW"]
    assert out["days"].tolist() == [14, 14]
    assert out["days"].dtype == "int64" or str(out["days"].dtype).startswith("int")
    assert pd.isna(out["forward_price"]).all()


def test_canonicalize_fills_optional_columns_and_infers_settlement() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "expiration": ["2021-08-04", "2021-08-04"],
            "cp_flag": ["c", "p"],
            "strike": [4000.0, 3900.0],
            "Bid": [10.0, 11.5],
            "Ask": [10.2, 11.7],
            "symbol": ["SPX 210804C4000", "SPXW 210804P3900"],
        }
    )
    out = canonicalize(
        raw,
        ColumnMap.detect(raw.columns),
        strike_scale=1.0,
        dayfirst=None,
        cp_values=None,
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert pd.isna(out["volume"]).all()
    assert pd.isna(out["open_interest"]).all()
    assert pd.isna(out["impl_volatility"]).all()
    assert out["quote_id"].tolist() == ["r0", "r1"]
    assert out["am_settlement"].tolist() == [1, 0]
    assert out["expiry_indicator"].tolist() == ["", ""]
    assert out["root"].tolist() == ["SPX", "SPXW"]


def test_canonicalize_drops_rows_missing_required_values() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21"],
            "exdate": ["2021-08-04", "2021-08-04"],
            "cp_flag": ["C", "P"],
            "strike": [4000.0, 3900.0],
            "bid": [10.0, pd.NA],
            "ask": [10.2, 11.7],
        }
    )
    out = canonicalize(
        raw,
        ColumnMap.detect(raw.columns),
        strike_scale=1.0,
        dayfirst=None,
        cp_values=None,
    )
    assert len(out) == 1
    assert out["cp_flag"].tolist() == ["C"]


def test_canonical_and_required_fields_match_spec() -> None:
    assert CANONICAL_FIELDS[0] == "date"
    assert REQUIRED_FIELDS == ("date", "expiry", "cp_flag", "strike", "bid", "ask")
