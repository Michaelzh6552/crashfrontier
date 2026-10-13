from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crashfrontier.data.forwards import (
    ForwardColumn,
    ForwardTable,
    ParityForward,
    resolve_forward,
)
from crashfrontier.data.rates import (
    ConstantRate,
    SeriesRate,
    ZeroCurveRate,
    ZeroRate,
    open_rate_source,
)
from crashfrontier.data.spot import SpotTable
from crashfrontier.errors import MissingForwardError, SchemaError
from tests.synthetic import standard_market


def _forward_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21", "2021-07-21", "2021-07-21"],
            "expiration": ["2021-08-04", "2021-08-20", "2021-08-20", "2021-08-20"],
            "AMSettlement": [0, 0, 1, 0],
            "ForwardPrice": [4010.0, 4020.5, 4021.0, 4019.0],
        }
    )


def test_forward_table_three_key_lookup() -> None:
    table = ForwardTable.from_frame(_forward_rows())
    assert table.name == "table"
    assert table.forward("2021-07-21", "2021-08-20", 1) == pytest.approx(4021.0)
    assert table.forward("2021-07-21", "2021-08-20", 0) == pytest.approx(4019.0)
    assert table.forward("2021-07-21", "2021-08-04", 0) == pytest.approx(4010.0)


def test_forward_spot_rate_queries_drop_timestamp_timezone() -> None:
    aware = pd.Timestamp("2021-07-21 15:00", tz="UTC")
    expiry = pd.Timestamp("2021-08-20 09:30", tz="America/New_York")
    table = ForwardTable.from_frame(_forward_rows())
    assert table.forward(aware, expiry, 1) == pytest.approx(4021.0)
    assert table.prices_on(aware)
    spot = SpotTable(pd.DataFrame({"date": ["2021-07-21"], "close": [4000.0]}))
    assert spot.spot(aware) == pytest.approx(4000.0)
    series = SeriesRate(
        pd.DataFrame({"observation_date": ["2021-07-21"], "DGS1MO": [5.0]}),
        unit="percent",
    )
    assert series.rate(aware, 30) == pytest.approx(0.05)
    curve = ZeroCurveRate(
        pd.DataFrame(
            {"date": ["2021-07-21", "2021-07-21"], "days": [7, 365], "rate": [5.0, 7.0]}
        )
    )
    assert curve.rate(aware, 7) == pytest.approx(0.05)


def test_forward_table_two_key_fallback() -> None:
    table = ForwardTable.from_frame(_forward_rows())
    assert table.forward("2021-07-21", "2021-08-04", 1) == pytest.approx(4010.0)
    assert table.forward("2021-07-22", "2021-08-20", 0) is None


def test_forward_table_ambiguous_uses_pm() -> None:
    table = ForwardTable.from_frame(_forward_rows())
    assert table.forward("2021-07-21", "2021-08-20", None) == pytest.approx(4019.0)


def test_forward_table_duplicate_keeps_last() -> None:
    rows = pd.concat(
        [
            _forward_rows(),
            pd.DataFrame(
                {
                    "date": ["2021-07-21"],
                    "expiration": ["2021-08-04"],
                    "AMSettlement": [0],
                    "ForwardPrice": [4111.0],
                }
            ),
        ],
        ignore_index=True,
    )
    table = ForwardTable.from_frame(rows)
    assert table.forward("2021-07-21", "2021-08-04", 0) == pytest.approx(4111.0)


def test_forward_table_from_path(raw_dir: Path) -> None:
    table = ForwardTable.from_path(raw_dir / "forward.csv")
    _dates, _expiries, forwards, _rates = standard_market()
    date = pd.Timestamp("2021-07-21")
    expiry_14 = pd.Timestamp("2021-08-04")
    expiry_30 = pd.Timestamp("2021-08-20")
    assert table.forward(date, expiry_14, 0) == pytest.approx(forwards[(date, expiry_14, 0)])
    assert table.forward(date, expiry_30, 1) == pytest.approx(forwards[(date, expiry_30, 1)])
    assert table.forward(date, expiry_30, None) == pytest.approx(forwards[(date, expiry_30, 0)])


def test_forward_column_median_and_settlement() -> None:
    rows = pd.DataFrame(
        {
            "expiry": ["2021-08-04", "2021-08-04", "2021-08-04", "2021-08-20"],
            "am_settlement": [0, 0, 1, 0],
            "forward_price": [100.0, 102.0, 200.0, np.nan],
        }
    )
    column = ForwardColumn(rows)
    assert column.name == "column"
    assert column.forward("2021-07-21", "2021-08-04", 0) == pytest.approx(101.0)
    assert column.forward("2021-07-21", "2021-08-04", 1) == pytest.approx(200.0)
    assert column.forward("2021-07-21", "2021-08-04", None) == pytest.approx(101.0)
    assert column.forward("2021-07-21", "2021-08-20", 0) is None


def test_forward_column_two_key_when_one_settlement() -> None:
    rows = pd.DataFrame(
        {
            "expiry": ["2021-08-04"],
            "am_settlement": [0],
            "forward_price": [99.5],
        }
    )
    column = ForwardColumn(rows)
    assert column.forward("2021-07-21", "2021-08-04", 1) == pytest.approx(99.5)


def _parity_quotes(forward: float, discount: float, strikes: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for strike in strikes:
        gap = discount * (forward - float(strike))
        call_mid = 80.0 + max(gap, 0.0)
        put_mid = call_mid - gap
        for cp_flag, mid in (("C", call_mid), ("P", put_mid)):
            records.append(
                {"strike": float(strike), "cp_flag": cp_flag, "bid": mid, "ask": mid}
            )
    return pd.DataFrame.from_records(records)


def test_parity_forward_recovers_forward() -> None:
    forward = 4321.5
    discount = 0.998
    quotes = _parity_quotes(forward, discount, np.array([4000.0, 4100.0, 4200.0, 4300.0, 4400.0]))
    source = ParityForward(quotes, discount)
    inferred = source.forward("2021-07-21", "2021-08-20", 0)
    assert source.name == "parity"
    assert inferred is not None
    assert abs(inferred / forward - 1.0) < 1e-8


def test_parity_forward_too_few_strikes_returns_none() -> None:
    quotes = _parity_quotes(4000.0, 0.999, np.array([4000.0]))
    assert ParityForward(quotes, 0.999).forward("2021-07-21", "2021-08-20", 0) is None


def test_parity_forward_uses_nearest_to_spot() -> None:
    discount = 0.999
    true_forward = 4000.0
    clean = np.array([3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0, 4400.0])
    dirty = np.array([1000.0, 2000.0, 5000.0, 6000.0, 8000.0])
    quotes = pd.concat(
        [
            _parity_quotes(true_forward, discount, clean),
            _parity_quotes(9999.0, discount, dirty),
        ],
        ignore_index=True,
    )
    inferred = ParityForward(quotes, discount, spot=4000.0).forward(
        "2021-07-21", "2021-08-20", 0
    )
    assert inferred is not None
    assert abs(inferred / true_forward - 1.0) < 1e-8


def test_parity_forward_rejects_crossed_quotes() -> None:
    quotes = pd.DataFrame(
        {
            "strike": [3900.0, 3900.0, 4100.0, 4100.0],
            "cp_flag": ["C", "P", "C", "P"],
            "bid": [120.0, 10.0, 5.0, 80.0],
            "ask": [110.0, 12.0, 6.0, 82.0],
        }
    )
    assert ParityForward(quotes, 0.999).forward("2021-07-21", "2021-08-20", 0) is None


def test_resolve_forward_chain_and_missing() -> None:
    table = ForwardTable.from_frame(_forward_rows())
    column = ForwardColumn(
        pd.DataFrame(
            {
                "expiry": ["2021-09-01"],
                "am_settlement": [0],
                "forward_price": [4100.0],
            }
        )
    )
    value, name = resolve_forward(
        [table, column], "2021-07-21", "2021-08-20", 1
    )
    assert name == "table"
    assert value == pytest.approx(4021.0)
    value, name = resolve_forward(
        [table, column], "2021-07-21", "2021-09-01", 0
    )
    assert name == "column"
    assert value == pytest.approx(4100.0)
    with pytest.raises(MissingForwardError, match="tried"):
        resolve_forward([table, column], "2021-07-21", "2021-10-01", 0)


def test_series_rate_ffill_percent_and_decimal() -> None:
    table = pd.DataFrame(
        {
            "observation_date": ["2021-07-19", "2021-07-20", "2021-07-21"],
            "DGS1MO": [5.0, np.nan, 6.0],
        }
    )
    percent = SeriesRate(table, unit="percent")
    assert percent.name == "series"
    assert percent.rate("2021-07-19", 30) == pytest.approx(0.05)
    assert percent.rate("2021-07-20", 30) == pytest.approx(0.05)
    assert percent.rate("2021-07-21", 7) == pytest.approx(0.06)
    decimal = SeriesRate(table, unit="decimal")
    assert decimal.rate("2021-07-20", 30) == pytest.approx(5.0)
    assert decimal.rate("2021-07-21", 30) == pytest.approx(6.0)


def test_series_rate_before_first_raises() -> None:
    table = pd.DataFrame({"observation_date": ["2021-07-21"], "DGS1MO": [5.0]})
    source = SeriesRate(table, unit="percent")
    with pytest.raises(ValueError, match="rates=0.0"):
        source.rate("2021-07-20", 30)


def test_series_rate_date_fallback_column() -> None:
    table = pd.DataFrame({"asof": ["2021-07-21", "2021-07-22"], "rf": [4.0, 5.0]})
    source = SeriesRate(table, unit="percent")
    assert source.rate("2021-07-22", 14) == pytest.approx(0.05)


def test_zero_curve_interpolation_and_extrapolation() -> None:
    table = pd.DataFrame(
        {
            "date": ["2021-07-21", "2021-07-21", "2021-07-28", "2021-07-28"],
            "days": [7, 365, 7, 365],
            "rate": [5.0, 7.0, 6.0, 8.0],
        }
    )
    curve = ZeroCurveRate(table)
    assert curve.name == "zero_curve"
    expected = 0.05 + (0.07 - 0.05) * (30.0 - 7.0) / (365.0 - 7.0)
    assert curve.rate("2021-07-21", 30) == pytest.approx(expected)
    assert curve.rate("2021-07-21", 1) == pytest.approx(0.05)
    assert curve.rate("2021-07-21", 400) == pytest.approx(0.07)
    assert curve.rate("2021-07-25", 7) == pytest.approx(0.05)
    assert curve.rate("2021-07-28", 7) == pytest.approx(0.06)
    with pytest.raises(ValueError, match="zero curve"):
        curve.rate("2021-07-20", 30)


def test_constant_rate() -> None:
    source = ConstantRate(0.053)
    assert source.name == "constant"
    assert source.rate("2013-01-02", 30) == pytest.approx(0.053)
    assert source.rate("2025-08-29", 7) == pytest.approx(0.053)


def test_zero_rate_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    source = ZeroRate()
    caplog.set_level(logging.WARNING, logger="crashfrontier")
    assert source.name == "zero"
    assert source.rate("2021-07-21", 30) == 0.0
    assert source.rate("2021-07-22", 7) == 0.0
    records = [record for record in caplog.records if record.name == "crashfrontier"]
    assert len(records) == 1
    assert "r = 0" in records[0].message


def test_open_rate_source_dispatch(tmp_path: Path) -> None:
    assert isinstance(open_rate_source(None), ZeroRate)
    constant = open_rate_source(0.02)
    assert isinstance(constant, ConstantRate)
    assert constant.rate("2021-07-21", 30) == pytest.approx(0.02)
    series_path = tmp_path / "DGS1MO.csv"
    pd.DataFrame(
        {"observation_date": ["2021-07-21"], "DGS1MO": [5.0]}
    ).to_csv(series_path, index=False)
    series = open_rate_source(series_path, unit="percent")
    assert isinstance(series, SeriesRate)
    assert series.rate("2021-07-21", 30) == pytest.approx(0.05)
    curve_frame = pd.DataFrame(
        {"date": ["2021-07-21", "2021-07-21"], "days": [7, 365], "zero_rate": [5.0, 5.0]}
    )
    curve = open_rate_source(curve_frame)
    assert isinstance(curve, ZeroCurveRate)
    assert curve.rate("2021-07-21", 30) == pytest.approx(0.05)


def test_spot_table_aliases_thousands_and_bom(raw_dir: Path) -> None:
    table = SpotTable.from_path(raw_dir / "SPX.csv")
    assert table.spot("2021-07-21") == pytest.approx(4000.0)
    assert np.isnan(table.spot("2021-07-23"))
    close = SpotTable(
        pd.DataFrame({"Date": ["07/21/2021"], "Close": ["4,000.25"]})
    )
    assert close.spot("2021-07-21") == pytest.approx(4000.25)
    adj = SpotTable(
        pd.DataFrame({"date": ["2021-07-21"], "Adj Close": [3999.5]})
    )
    assert adj.spot(pd.Timestamp("2021-07-21")) == pytest.approx(3999.5)


def test_spot_table_missing_columns_raise() -> None:
    with pytest.raises(SchemaError, match="spot"):
        SpotTable(pd.DataFrame({"date": ["2021-07-21"], "volume": [1.0]}))


def test_forward_table_missing_columns_raise() -> None:
    with pytest.raises(SchemaError, match="forward"):
        ForwardTable.from_frame(pd.DataFrame({"date": ["2021-07-21"], "expiration": ["2021-08-20"]}))
