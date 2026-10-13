from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic import (
    OPTIONMETRICS_COLUMNS,
    QUOTE_COLUMNS,
    FrameBundle,
    black76,
    scenarios,
    write_optionmetrics_dir,
)


@pytest.mark.parametrize(
    ("forward", "strike", "tau", "sigma", "discount"),
    [
        (4000.0, 3900.0, 14.0 / 365.0, 0.18, 0.999),
        (4000.0, 4000.0, 7.0 / 365.0, 0.18, 1.0),
        (4000.0, 4200.0, 30.0 / 365.0, 0.25, 0.995),
        (4008.25, 2800.0, 30.0 / 365.0, 0.18, 0.9995),
    ],
)
def test_black76_put_call_parity(
    forward: float, strike: float, tau: float, sigma: float, discount: float
) -> None:
    call = black76(forward, strike, tau, sigma, discount, "C")
    put = black76(forward, strike, tau, sigma, discount, "P")
    assert call - put == pytest.approx(discount * (forward - strike), rel=0.0, abs=1e-10)
    assert call == pytest.approx(black76(forward, strike, tau, sigma, discount, "c"))
    assert put == pytest.approx(black76(forward, strike, tau, sigma, discount, "put"))


def test_raw_dir_files_rows_and_columns(raw_dir: Path) -> None:
    option_path = raw_dir / "option_price.csv"
    forward_path = raw_dir / "forward.csv"
    rate_path = raw_dir / "DGS1MO.csv"
    spot_path = raw_dir / "SPX.csv"
    vix_path = raw_dir / "VIX_History.csv"
    assert option_path.is_file()
    assert forward_path.is_file()
    assert rate_path.is_file()
    assert spot_path.is_file()
    assert vix_path.is_file()

    options = pd.read_csv(option_path)
    assert list(options.columns) == list(OPTIONMETRICS_COLUMNS)
    assert len(options.columns) == 38
    assert len(options) == 600
    dates = pd.to_datetime(options["date"])
    assert set(dates.dt.strftime("%Y-%m-%d")) == {"2021-07-21", "2021-07-22", "2021-07-28"}
    first_day = options.loc[dates.eq(pd.Timestamp("2021-07-21"))]
    assert first_day.groupby(["exdate", "am_settlement"]).ngroups == 4
    assert (first_day.groupby(["exdate", "am_settlement"]).size() == 50).all()
    assert options["optionid"].nunique() == len(options)
    assert options["forward_price"].isna().all()
    assert options["strike_price"].median() == pytest.approx(4_000_000.0, rel=0.15)

    forwards = pd.read_csv(forward_path)
    assert {"date", "expiration", "AMSettlement", "ForwardPrice"} <= set(forwards.columns)
    assert len(forwards) == 12
    assert set(forwards["AMSettlement"]) == {0, 1}

    rates = pd.read_csv(rate_path)
    assert list(rates.columns) == ["observation_date", "DGS1MO"]
    assert rates["DGS1MO"].isna().sum() == 1

    spot_bytes = spot_path.read_bytes()
    assert spot_bytes.startswith(b"\xef\xbb\xbf")
    spot_text = spot_bytes.decode("utf-8-sig")
    assert '"Date","Price"' in spot_text.splitlines()[0]
    assert "07/21/2021" in spot_text
    assert "4,000.00" in spot_text

    vix = pd.read_csv(vix_path)
    assert len(vix) == 3


def test_scenarios_weights_normalized() -> None:
    expected = {"atom_at_k", "atom_below_k", "atom_above_k", "far_tail", "bimodal"}
    table = scenarios(0.9)
    assert set(table) == expected
    for name, (nodes, weights) in table.items():
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0.0).all()
        assert nodes.shape == weights.shape
        mean = float((nodes * weights).sum())
        assert mean == pytest.approx(1.0, abs=1e-12), name


def test_atomic_quotes_fixture(atomic_quotes: pd.DataFrame) -> None:
    assert list(atomic_quotes.columns) == list(QUOTE_COLUMNS)
    assert (atomic_quotes["cp_flag"] == "P").all()
    assert (atomic_quotes["norm_l"] <= atomic_quotes["norm_u"]).all()
    assert atomic_quotes["quote_id"].tolist()[0] == "syn0"
    assert (atomic_quotes["x"] == atomic_quotes["strike"]).all()


def test_variant_dir_layouts(variant_dir: Path | FrameBundle, request: pytest.FixtureRequest) -> None:
    name = request.node.callspec.params["variant_dir"]
    if name == "frames":
        assert isinstance(variant_dir, FrameBundle)
        assert len(variant_dir.options) == 600
        assert variant_dir.forwards is not None
        assert variant_dir.rates is not None
        assert variant_dir.spot is not None
        return

    root = Path(variant_dir)
    names = {path.name for path in root.iterdir()}
    assert "VIX_History.csv" in names
    if name == "uppercase_semicolon_mdy":
        assert "OPTIONS.txt" in names
        assert "option_price.csv" not in names
        assert "forward.csv" in names
        header = (root / "OPTIONS.txt").read_text(encoding="utf-8").splitlines()[0]
        assert ";" in header
        assert "DATE" in header
    elif name == "yyyymmdd_gz_column_forward":
        assert "option_price.csv.gz" in names
        assert "forward.csv" not in names
        assert "zerocd.csv" in names
        curve = pd.read_csv(root / "zerocd.csv")
        assert list(curve.columns) == ["date", "days", "rate"]
        assert set(curve["days"]) == {7, 365}
    elif name == "date9_split_dollar_novolume":
        assert {"opt_2021a.csv", "opt_2021b.csv"} <= names
        assert "option_price.csv" not in names
        first = pd.read_csv(root / "opt_2021a.csv")
        assert "volume" not in first.columns
        assert "symbol" not in first.columns
        assert "am_settlement" not in first.columns
        assert first["date"].astype(str).str.fullmatch(r"\d{2}[A-Z]{3}\d{4}").all()
        assert first["strike_price"].median() < 20_000
        forwards = pd.read_csv(root / "forward.csv")
        assert set(forwards["AMSettlement"]) == {0}
    elif name == "stata":
        assert {"option_price.dta", "forward.dta", "DGS1MO.dta", "SPX.dta"} <= names
    elif name == "parquet":
        assert {
            "option_price.parquet",
            "forward.parquet",
            "DGS1MO.parquet",
            "SPX.parquet",
        } <= names
    else:
        raise AssertionError(name)


def test_write_optionmetrics_dir_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        write_optionmetrics_dir(tmp_path, name="not_a_variant")
