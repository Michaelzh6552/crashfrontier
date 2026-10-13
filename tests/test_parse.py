from __future__ import annotations

import logging

import pandas as pd
import pytest

from crashfrontier.data.parse import (
    DATE_FORMATS,
    detect_date_format,
    detect_strike_scale,
    parse_cp_flag,
    parse_dates,
    parse_numeric,
    parse_settlement,
)
from crashfrontier.errors import SchemaError

STRING_FORMAT_EXAMPLES = [
    ("%Y-%m-%d", "2021-07-21"),
    ("%Y%m%d", "20210721"),
    ("%m/%d/%Y", "07/21/2021"),
    ("%d/%m/%Y", "21/07/2021"),
    ("%Y/%m/%d", "2021/07/21"),
    ("%d%b%Y", "21JUL2021"),
    ("%d-%b-%Y", "21-JUL-2021"),
    ("%d-%b-%y", "21-Jul-21"),
    ("%b %d, %Y", "Jul 21, 2021"),
    ("%Y-%m-%d %H:%M:%S", "2021-07-21 15:30:00"),
    ("%Y-%m-%dT%H:%M:%S", "2021-07-21T15:30:00"),
    ("%m/%d/%y", "07/21/21"),
    ("%d.%m.%Y", "21.07.2021"),
    ("%Y.%m.%d", "2021.07.21"),
    ("%m-%d-%Y", "07-21-2021"),
    ("%d-%m-%Y", "21-07-2021"),
]


@pytest.mark.parametrize("fmt, raw", STRING_FORMAT_EXAMPLES)
def test_parse_dates_each_string_format(fmt: str, raw: str) -> None:
    result = parse_dates(pd.Series([raw]))
    assert result.iloc[0] == pd.Timestamp("2021-07-21")
    assert str(result.dtype).startswith("datetime64")
    detected = detect_date_format(pd.Series([raw]))
    if raw.isdigit():
        assert detected == "%Y%m%d"
    else:
        assert detected == fmt


def test_date_formats_tuple_matches_spec() -> None:
    assert DATE_FORMATS == (
        "%Y-%m-%d",
        "%Y%m%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d%b%Y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%m-%d-%Y",
        "%d-%m-%Y",
    )


def test_parse_dates_numeric_yyyymmdd(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    result = parse_dates(pd.Series([20210721, 20210722]))
    assert result.tolist() == [pd.Timestamp("2021-07-21"), pd.Timestamp("2021-07-22")]
    assert "YYYYMMDD" in caplog.text


def test_parse_dates_yyyymmdd_decimal_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    objects = pd.Series(["20210721.0", "20210722.00"], dtype="object")
    strings = pd.Series(["20210721.0"], dtype="string")
    parsed = parse_dates(objects)
    assert parsed.tolist() == [pd.Timestamp("2021-07-21"), pd.Timestamp("2021-07-22")]
    assert parse_dates(strings).iloc[0] == pd.Timestamp("2021-07-21")
    assert detect_date_format(strings) == "%Y%m%d"
    assert "YYYYMMDD" in caplog.text


def test_parse_dates_sas_days(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    sas = (pd.Timestamp("2021-07-21") - pd.Timestamp("1960-01-01")).days
    result = parse_dates(pd.Series([sas]))
    assert result.iloc[0] == pd.Timestamp("2021-07-21")
    assert 5000 <= sas <= 30000
    assert "SAS" in caplog.text
    assert detect_date_format(pd.Series([sas])) == "sas_days"


def test_parse_dates_excel_serial(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    excel = (pd.Timestamp("2021-07-21") - pd.Timestamp("1899-12-30")).days
    result = parse_dates(pd.Series([excel]))
    assert result.iloc[0] == pd.Timestamp("2021-07-21")
    assert 30000 <= excel <= 80000
    assert "Excel" in caplog.text
    assert detect_date_format(pd.Series([excel])) == "excel_serial"


def test_parse_dates_datetime64_normalizes_and_drops_tz() -> None:
    aware = pd.Series(pd.to_datetime(["2021-07-21 15:30:00"]).tz_localize("UTC"))
    result = parse_dates(aware)
    assert result.iloc[0] == pd.Timestamp("2021-07-21")
    assert result.dt.tz is None


def test_parse_dates_unrecognized_numeric_raises() -> None:
    with pytest.raises(SchemaError, match="unrecognized numeric date"):
        parse_dates(pd.Series([123]))


def test_parse_dates_dayfirst_explicit() -> None:
    values = pd.Series(["05/06/2021"])
    assert parse_dates(values, dayfirst=True).iloc[0] == pd.Timestamp("2021-06-05")
    assert parse_dates(values, dayfirst=False).iloc[0] == pd.Timestamp("2021-05-06")
    assert parse_dates(pd.Series(["05-06-2021"]), dayfirst=True).iloc[0] == pd.Timestamp("2021-06-05")


def test_parse_dates_dayfirst_inferred_from_segments() -> None:
    day_first = parse_dates(pd.Series(["13/05/2021", "14/05/2021"]))
    assert day_first.tolist() == [pd.Timestamp("2021-05-13"), pd.Timestamp("2021-05-14")]
    month_first = parse_dates(pd.Series(["05/13/2021", "06/14/2021"]))
    assert month_first.tolist() == [pd.Timestamp("2021-05-13"), pd.Timestamp("2021-06-14")]


def test_parse_dates_ambiguous_defaults_to_month_first(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    result = parse_dates(pd.Series(["05/06/2021"]))
    assert result.iloc[0] == pd.Timestamp("2021-05-06")
    assert "assuming month-first" in caplog.text


def test_parse_numeric_thousands_and_dollar() -> None:
    values = pd.Series(["$1,234.50", "2,000", ".", "", "3.5"])
    result = parse_numeric(values)
    assert result.tolist()[0] == pytest.approx(1234.5)
    assert result.tolist()[1] == pytest.approx(2000.0)
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])
    assert result.tolist()[4] == pytest.approx(3.5)


def test_parse_numeric_numeric_dtype() -> None:
    result = parse_numeric(pd.Series([1, 2], dtype="int64"))
    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 2.0]


def test_parse_cp_flag_call_put_letters() -> None:
    values = pd.Series(["Call", "Put", "c", "p", "C", "P"])
    assert parse_cp_flag(values).tolist() == ["C", "P", "C", "P", "C", "P"]


def test_parse_cp_flag_numeric_with_cp_values() -> None:
    values = pd.Series(["1", "0", 1, 0])
    result = parse_cp_flag(values, cp_values={"1": "C", "0": "P"})
    assert result.tolist() == ["C", "P", "C", "P"]


def test_parse_cp_flag_unknown_raises() -> None:
    with pytest.raises(SchemaError, match="unrecognized cp_flag"):
        parse_cp_flag(pd.Series(["X"]))


def test_parse_settlement_numeric_and_strings() -> None:
    root = pd.Series(["SPXW", "SPXW", "SPXW", "SPXW", "SPXW", "SPXW"])
    numeric = parse_settlement(pd.Series([1, 0, 2, 0, 1, 0]), root)
    assert numeric.tolist() == [1, 0, 1, 0, 1, 0]
    labels = parse_settlement(pd.Series(["AM", "PM", "A", "P", "TRUE", "N"]), root)
    assert labels.tolist() == [1, 0, 1, 0, 1, 0]
    flags = parse_settlement(pd.Series(["Y", "FALSE", "1", "0", "am", "pm"]), root)
    assert flags.tolist() == [1, 0, 1, 0, 1, 0]


def test_parse_settlement_missing_column_uses_root() -> None:
    root = pd.Series(["SPX", "SPXW", "", "SPX"])
    result = parse_settlement(None, root)
    assert result.tolist() == [1, 0, 0, 1]


def test_parse_settlement_unknown_raises() -> None:
    with pytest.raises(SchemaError, match="unrecognized settlement"):
        parse_settlement(pd.Series(["WEEKLY"]), pd.Series(["SPX"]))


def test_detect_strike_scale_with_reference_price() -> None:
    scaled = pd.Series([3_900_000.0, 4_000_000.0, 4_100_000.0])
    assert detect_strike_scale(scaled, 4000.0) == 1000.0
    native = pd.Series([3900.0, 4000.0, 4100.0])
    assert detect_strike_scale(native, 4000.0) == 1.0
    tens = pd.Series([39000.0, 40000.0, 41000.0])
    assert detect_strike_scale(tens, 4000.0) == 10.0
    hundreds = pd.Series([390000.0, 400000.0, 410000.0])
    assert detect_strike_scale(hundreds, 4000.0) == 100.0


def test_detect_strike_scale_without_reference(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    assert detect_strike_scale(pd.Series([4_000_000.0]), float("nan")) == 1000.0
    assert detect_strike_scale(pd.Series([4000.0]), float("nan")) == 1.0
    assert "pass strike_scale explicitly" in caplog.text
