from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import ndtr

__all__ = [
    "FRAME_VARIANT",
    "FrameBundle",
    "OPTIONMETRICS_COLUMNS",
    "QUOTE_COLUMNS",
    "VARIANT_NAMES",
    "black76",
    "make_forward_table",
    "make_option_rows",
    "scenarios",
    "standard_market",
    "synthetic_quotes",
    "true_targets",
    "write_optionmetrics_dir",
]

OPTIONMETRICS_COLUMNS = (
    "secid",
    "date",
    "symbol",
    "symbol_flag",
    "exdate",
    "last_date",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
    "cfadj",
    "am_settlement",
    "contract_size",
    "ss_flag",
    "forward_price",
    "expiry_indicator",
    "root",
    "suffix",
    "cusip",
    "ticker",
    "sic",
    "index_flag",
    "exchange_d",
    "class",
    "issue_type",
    "industry_group",
    "issuer",
    "div_convention",
    "exercise_style",
    "am_set_flag",
)
QUOTE_COLUMNS = (
    "quote_id",
    "symbol",
    "root",
    "cp_flag",
    "strike",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "impl_volatility",
    "x",
    "log_moneyness",
    "norm_bid",
    "norm_ask",
    "norm_l",
    "norm_u",
    "role",
)
VARIANT_NAMES = (
    "uppercase_semicolon_mdy",
    "yyyymmdd_gz_column_forward",
    "date9_split_dollar_novolume",
    "stata",
    "parquet",
    "frames",
)
FRAME_VARIANT = "frames"

_SECID = 108105
_CUSIP = 64881510
_UNDERLYING = "SPX"
_WEEKLY_ROOT = "SPXW"
_CONTRACT_SIZE = 100
_EXCHANGE = 32768
_DEFAULT_SIGMA = 0.18
_DEFAULT_N_STRIKES = 25
_STRIKE_MONEYNESS_LO = 0.70
_STRIKE_MONEYNESS_HI = 1.20
_STRIKE_ROUND = 5.0
_HALF_SPREAD = 0.0005
_VOLUME_MONEYNESS = 0.75
_STRIKE_SCALE = 1000.0
_DAYS_PER_YEAR = 365.0
_AM_FORWARD_GAP = 0.05
_FIRST_OPTION_ID = 1_000_001
_TENORS = (7, 14, 30)
_DATE_STRINGS = ("2021-07-21", "2021-07-22", "2021-07-28")
_SPOT_LEVELS = {
    "2021-07-19": 3992.0,
    "2021-07-20": 3996.5,
    "2021-07-21": 4000.0,
    "2021-07-22": 4008.25,
    "2021-07-27": 4025.75,
    "2021-07-28": 4032.5,
}
_RATE_PERCENT = {
    "2021-07-19": 0.05,
    "2021-07-20": None,
    "2021-07-21": 0.06,
    "2021-07-22": 0.06,
    "2021-07-27": 0.07,
    "2021-07-28": 0.07,
}
_VIX_LEVELS = {
    "2021-07-21": 17.01,
    "2021-07-22": 16.84,
    "2021-07-28": 18.22,
}
_ZERO_CURVE_DAYS = (7, 365)
_SPLIT_DATE = pd.Timestamp("2021-07-28")


@dataclass(frozen=True)
class FrameBundle:
    """In-memory tables for ``Dataset.from_frames`` (the ``frames`` variant).

    File-based variants return a directory ``Path``. T08 should build a dataset
    with ``Dataset.from_frames(bundle.options, forwards=bundle.forwards,
    rates=bundle.rates, spot=bundle.spot)``.
    """

    options: pd.DataFrame
    forwards: pd.DataFrame | None
    rates: pd.DataFrame | None
    spot: pd.DataFrame | None


def black76(forward, strike, tau, sigma, discount, cp_flag) -> float:
    """Black-76 undiscretized option price."""
    flag = str(cp_flag).strip().upper()[:1]
    if flag not in {"C", "P"}:
        raise ValueError(f"unrecognized cp_flag {cp_flag!r}")
    forward = float(forward)
    strike = float(strike)
    tau = float(tau)
    sigma = float(sigma)
    discount = float(discount)
    if tau <= 0.0 or sigma <= 0.0:
        intrinsic = max(forward - strike, 0.0) if flag == "C" else max(strike - forward, 0.0)
        return discount * intrinsic
    # d1 = (ln(F/κ) + σ²τ/2) / (σ√τ), d2 = d1 - σ√τ
    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if flag == "C":
        return discount * (forward * float(ndtr(d1)) - strike * float(ndtr(d2)))
    return discount * (strike * float(ndtr(-d2)) - forward * float(ndtr(-d1)))


def standard_market() -> tuple[
    list[pd.Timestamp],
    dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]],
    dict[tuple[pd.Timestamp, pd.Timestamp, int], float],
    dict[pd.Timestamp, float],
]:
    """Default three-day calendar: tenors 7/14/30, AM+PM on the 30-day expiry."""
    dates = [_as_timestamp(value) for value in _DATE_STRINGS]
    expiries_by_date: dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]] = {}
    forwards: dict[tuple[pd.Timestamp, pd.Timestamp, int], float] = {}
    rates = {_as_timestamp(date): percent / 100.0 for date, percent in _RATE_PERCENT.items() if percent is not None}
    for date in dates:
        groups: list[tuple[pd.Timestamp, int]] = []
        spot = _SPOT_LEVELS[_iso(date)]
        rate = rates[date]
        for days in _TENORS:
            expiry = date + pd.Timedelta(days=days)
            settlements = (0, 1) if days == 30 else (0,)
            for am_settlement in settlements:
                forward = spot * math.exp(rate * days / _DAYS_PER_YEAR)
                if am_settlement == 1:
                    forward += _AM_FORWARD_GAP
                forwards[(date, expiry, am_settlement)] = float(forward)
                groups.append((expiry, am_settlement))
        expiries_by_date[date] = groups
    return dates, expiries_by_date, forwards, {date: rates[date] for date in dates}


def make_option_rows(
    dates,
    expiries_by_date,
    forwards,
    rates,
    *,
    sigma: float = _DEFAULT_SIGMA,
    n_strikes: int = _DEFAULT_N_STRIKES,
) -> pd.DataFrame:
    """OptionMetrics-format option table priced with Black-76."""
    date_list = [_as_timestamp(value) for value in dates]
    expiry_map = _as_expiry_map(expiries_by_date)
    forward_map = _as_forward_map(forwards)
    rate_map = _as_rate_map(rates)
    records: list[dict[str, object]] = []
    option_id = _FIRST_OPTION_ID
    for date in date_list:
        rate = rate_map[date]
        for expiry, am_settlement in expiry_map[date]:
            days = int((expiry - date).days)
            tau = days / _DAYS_PER_YEAR
            discount = math.exp(-rate * tau)
            forward = forward_map[(date, expiry, am_settlement)]
            root = _UNDERLYING if am_settlement == 1 else _WEEKLY_ROOT
            expiry_indicator = "" if am_settlement == 1 else "w"
            yymmdd = expiry.strftime("%y%m%d")
            for strike in _strike_grid(forward, n_strikes):
                strike_price = int(round(strike * _STRIKE_SCALE))
                half_spread = _HALF_SPREAD * forward
                volume = 10 if strike >= _VOLUME_MONEYNESS * forward else 0
                open_interest = 100 if volume else 0
                for cp_flag in ("C", "P"):
                    price = black76(forward, strike, tau, sigma, discount, cp_flag)
                    bid = max(price - half_spread, 0.0)
                    ask = price + half_spread
                    records.append(
                        {
                            "secid": _SECID,
                            "date": _iso(date),
                            "symbol": f"{root} {yymmdd}{cp_flag}{strike_price}",
                            "symbol_flag": 1,
                            "exdate": _iso(expiry),
                            "last_date": "",
                            "cp_flag": cp_flag,
                            "strike_price": strike_price,
                            "best_bid": bid,
                            "best_offer": ask,
                            "volume": volume,
                            "open_interest": open_interest,
                            "impl_volatility": sigma,
                            "delta": np.nan,
                            "gamma": np.nan,
                            "vega": np.nan,
                            "theta": np.nan,
                            "optionid": option_id,
                            "cfadj": 1,
                            "am_settlement": am_settlement,
                            "contract_size": _CONTRACT_SIZE,
                            "ss_flag": 0,
                            "forward_price": np.nan,
                            "expiry_indicator": expiry_indicator,
                            "root": "",
                            "suffix": "",
                            "cusip": _CUSIP,
                            "ticker": _UNDERLYING,
                            "sic": 9999,
                            "index_flag": 1,
                            "exchange_d": _EXCHANGE,
                            "class": "",
                            "issue_type": "A",
                            "industry_group": "",
                            "issuer": "CBOE S&P 500 INDEX",
                            "div_convention": "I",
                            "exercise_style": "E",
                            "am_set_flag": "",
                        }
                    )
                    option_id += 1
    frame = pd.DataFrame.from_records(records, columns=list(OPTIONMETRICS_COLUMNS))
    return frame.reset_index(drop=True)


def make_forward_table(
    forwards: Mapping[tuple[object, object, int], float],
) -> pd.DataFrame:
    """OptionMetrics-style forward table from a (date, expiry, AM) map."""
    records = [
        {
            "secid": _SECID,
            "date": _iso(date),
            "expiration": _iso(expiry),
            "AMSettlement": int(am_settlement),
            "ForwardPrice": float(price),
        }
        for (date, expiry, am_settlement), price in _as_forward_map(forwards).items()
    ]
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["date", "expiration", "AMSettlement"], kind="mergesort")
    return frame.reset_index(drop=True)


def synthetic_quotes(nodes, weights, strikes=None, spread: float = 0.0) -> pd.DataFrame:
    """Normalized put quotes generated from an atomic distribution."""
    node_values = np.asarray(nodes, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if strikes is None:
        strike_values = np.unique(np.r_[np.linspace(0.72, 1.18, 22), 0.85, 0.9])
    else:
        strike_values = np.unique(np.asarray(strikes, dtype=float))
    call = np.array(
        [float(np.sum(weight_values * np.maximum(node_values - strike, 0.0))) for strike in strike_values]
    )
    intrinsic = np.maximum(1.0 - strike_values, 0.0)
    norm_l = np.maximum(call - spread, intrinsic)
    norm_u = np.minimum(call + spread, 1.0)
    put_l = norm_l - (1.0 - strike_values)
    put_u = norm_u - (1.0 - strike_values)
    role = np.where(strike_values <= 1.0, "otm_put", "itm")
    frame = pd.DataFrame(
        {
            "quote_id": [f"syn{index}" for index in range(len(strike_values))],
            "symbol": "",
            "root": "",
            "cp_flag": "P",
            "strike": strike_values,
            "bid": put_l,
            "ask": put_u,
            "volume": 10.0,
            "open_interest": 10.0,
            "impl_volatility": np.nan,
            "x": strike_values,
            "log_moneyness": np.log(strike_values),
            "norm_bid": put_l,
            "norm_ask": put_u,
            "norm_l": norm_l,
            "norm_u": norm_u,
            "role": role,
        }
    )
    return frame.loc[:, list(QUOTE_COLUMNS)].reset_index(drop=True)


def scenarios(K) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named atomic distributions with mean one, parameterized by threshold K."""
    threshold = float(K)
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, location in (
        ("atom_at_k", threshold),
        ("atom_below_k", threshold - 1e-6),
        ("atom_above_k", threshold + 1e-6),
    ):
        nodes = np.array([location, 1.0, 2.0 - location], dtype=float)
        weights = np.array([0.15, 0.70, 0.15], dtype=float)
        table[name] = (nodes, weights)
    table["far_tail"] = (
        np.array([0.9, 1.0, 5.0], dtype=float),
        np.array([0.40, 0.59, 0.01], dtype=float),
    )
    table["bimodal"] = (
        np.array([0.82, 1.18], dtype=float),
        np.array([0.5, 0.5], dtype=float),
    )
    return table


def true_targets(nodes, weights, K, gamma: int = 0) -> tuple[float, float, float]:
    """Crash probability, loss, and depth under an atomic distribution."""
    node_values = np.asarray(nodes, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    threshold = float(K)
    below = node_values <= threshold
    if int(gamma) == 0:
        probability = float(np.sum(weight_values[below]))
        loss = float(np.sum(weight_values * np.maximum(threshold - node_values, 0.0)))
        depth = loss / probability if probability > 0.0 else float("nan")
        return probability, loss, depth
    power = float(gamma)
    mass = float(np.sum(weight_values * np.power(node_values, power)))
    area_gamma = float(np.sum(weight_values[below] * np.power(node_values[below], power)))
    area_next = float(np.sum(weight_values[below] * np.power(node_values[below], power + 1.0)))
    probability = area_gamma / mass if mass > 0.0 else float("nan")
    loss = (threshold * area_gamma - area_next) / mass if mass > 0.0 else float("nan")
    depth = threshold - area_next / area_gamma if area_gamma > 0.0 else float("nan")
    return probability, loss, depth


def write_optionmetrics_dir(tmp_path, **variant) -> Path | FrameBundle:
    """Write a miniature OptionMetrics directory, or return in-memory frames."""
    output_dir = Path(tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = variant.get("name", "standard")
    allowed = {"standard", *VARIANT_NAMES}
    if name not in allowed:
        raise ValueError(f"unknown variant {name!r}; expected one of {sorted(allowed)}")
    options, forwards, rates, spot = _standard_tables()
    if name == FRAME_VARIANT:
        return FrameBundle(options=options, forwards=forwards, rates=rates, spot=spot)
    _write_variant(output_dir, name, options, forwards, rates, spot)
    return output_dir


def _standard_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates, expiries_by_date, forwards, rates = standard_market()
    options = make_option_rows(dates, expiries_by_date, forwards, rates)
    forward_table = make_forward_table(forwards)
    rate_table = _rate_series_table()
    spot_table = _spot_table()
    return options, forward_table, rate_table, spot_table


def _write_variant(
    output_dir: Path,
    name: str,
    options: pd.DataFrame,
    forwards: pd.DataFrame,
    rates: pd.DataFrame,
    spot: pd.DataFrame,
) -> None:
    _write_vix(output_dir / "VIX_History.csv")
    if name == "standard":
        _write_csv(output_dir / "option_price.csv", options)
        _write_csv(output_dir / "forward.csv", forwards)
        _write_csv(output_dir / "DGS1MO.csv", rates)
        _write_spot_csv(output_dir / "SPX.csv", spot)
        return
    if name == "uppercase_semicolon_mdy":
        formatted = _format_option_dates(options, "mdy")
        formatted.columns = [str(column).upper() for column in formatted.columns]
        _write_csv(output_dir / "OPTIONS.txt", formatted, sep=";")
        _write_csv(output_dir / "forward.csv", forwards)
        _write_csv(output_dir / "DGS1MO.csv", rates)
        _write_spot_csv(output_dir / "SPX.csv", spot)
        return
    if name == "yyyymmdd_gz_column_forward":
        filled = _fill_forward_price(options, forwards)
        formatted = _format_option_dates(filled, "yyyymmdd")
        _write_csv(output_dir / "option_price.csv.gz", formatted, compression="gzip")
        _write_csv(output_dir / "zerocd.csv", _zero_curve_table())
        _write_spot_csv(output_dir / "SPX.csv", spot)
        return
    if name == "date9_split_dollar_novolume":
        converted = _date9_options(options)
        stamps = pd.to_datetime(converted["date"].astype(str).str.title(), format="%d%b%Y")
        first = converted.loc[stamps < _SPLIT_DATE].reset_index(drop=True)
        second = converted.loc[stamps >= _SPLIT_DATE].reset_index(drop=True)
        _write_csv(output_dir / "opt_2021a.csv", first)
        _write_csv(output_dir / "opt_2021b.csv", second)
        pm_forwards = forwards.loc[forwards["AMSettlement"].eq(0)].reset_index(drop=True)
        _write_csv(output_dir / "forward.csv", pm_forwards)
        _write_csv(output_dir / "DGS1MO.csv", rates)
        _write_spot_csv(output_dir / "SPX.csv", spot)
        return
    if name == "stata":
        _write_stata(output_dir / "option_price.dta", _with_datetime(options, ("date", "exdate")))
        _write_stata(output_dir / "forward.dta", _with_datetime(forwards, ("date", "expiration")))
        _write_stata(output_dir / "DGS1MO.dta", _with_datetime(rates, ("observation_date",)))
        _write_stata(output_dir / "SPX.dta", _with_datetime(spot, ("Date",)))
        return
    if name == "parquet":
        _write_parquet(output_dir / "option_price.parquet", _with_datetime(options, ("date", "exdate")))
        _write_parquet(output_dir / "forward.parquet", _with_datetime(forwards, ("date", "expiration")))
        _write_parquet(output_dir / "DGS1MO.parquet", _with_datetime(rates, ("observation_date",)))
        _write_parquet(output_dir / "SPX.parquet", _with_datetime(spot, ("Date",)))
        return
    raise ValueError(f"unhandled variant {name!r}")


def _date9_options(options: pd.DataFrame) -> pd.DataFrame:
    # Drop zero-volume rows so default min_volume filtering matches the standard variant.
    converted = options.loc[options["volume"] >= 1].copy()
    converted["strike_price"] = converted["strike_price"].astype(float) / _STRIKE_SCALE
    converted = converted.drop(columns=["volume", "symbol", "am_settlement"])
    return _format_option_dates(converted, "date9").reset_index(drop=True)


def _fill_forward_price(options: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (_as_timestamp(date), _as_timestamp(expiry), int(am_settlement)): float(price)
        for date, expiry, am_settlement, price in forwards[
            ["date", "expiration", "AMSettlement", "ForwardPrice"]
        ].itertuples(index=False, name=None)
    }
    filled = options.copy()
    filled["forward_price"] = [
        lookup[(_as_timestamp(date), _as_timestamp(expiry), int(am_settlement))]
        for date, expiry, am_settlement in options[["date", "exdate", "am_settlement"]].itertuples(
            index=False, name=None
        )
    ]
    return filled


def _format_option_dates(options: pd.DataFrame, style: str) -> pd.DataFrame:
    formatted = options.copy()
    for column in ("date", "exdate", "last_date"):
        if column not in formatted.columns:
            continue
        formatted[column] = _format_date_values(formatted[column], style)
    return formatted


def _format_date_values(values: pd.Series, style: str) -> pd.Series:
    stamps = pd.to_datetime(values, errors="coerce")
    missing = stamps.isna()
    if style == "mdy":
        text = stamps.dt.strftime("%m/%d/%Y")
        return text.mask(missing, "")
    if style == "date9":
        text = stamps.dt.strftime("%d%b%Y").str.upper()
        return text.mask(missing, "")
    if style == "yyyymmdd":
        numbers = pd.Series(pd.NA, index=values.index, dtype="Int64")
        numbers.loc[~missing] = stamps.loc[~missing].dt.strftime("%Y%m%d").astype(int)
        return numbers
    text = stamps.dt.strftime("%Y-%m-%d")
    return text.mask(missing, "")


def _rate_series_table() -> pd.DataFrame:
    rows = [
        {"observation_date": date, "DGS1MO": percent}
        for date, percent in _RATE_PERCENT.items()
    ]
    return pd.DataFrame.from_records(rows)


def _zero_curve_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, percent in _RATE_PERCENT.items():
        if percent is None:
            continue
        for days in _ZERO_CURVE_DAYS:
            rows.append({"date": date, "days": days, "rate": percent})
    return pd.DataFrame.from_records(rows)


def _spot_table() -> pd.DataFrame:
    rows = [
        {"Date": _as_timestamp(date), "Price": float(price)}
        for date, price in _SPOT_LEVELS.items()
    ]
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values("Date", ascending=False).reset_index(drop=True)


def _write_vix(path: Path) -> None:
    rows = [
        {"DATE": _as_timestamp(date).strftime("%m/%d/%Y"), "CLOSE": close}
        for date, close in _VIX_LEVELS.items()
    ]
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def _write_spot_csv(path: Path, spot: pd.DataFrame) -> None:
    lines = ['"Date","Price"']
    ordered = spot.sort_values("Date", ascending=False)
    for stamp, price in ordered[["Date", "Price"]].itertuples(index=False, name=None):
        date_text = pd.Timestamp(stamp).strftime("%m/%d/%Y")
        lines.append(f'"{date_text}","{float(price):,.2f}"')
    path.write_bytes(b"\xef\xbb\xbf" + "\n".join(lines).encode("utf-8") + b"\n")


def _write_csv(path: Path, frame: pd.DataFrame, *, sep: str = ",", compression: str | None = None) -> None:
    frame.to_csv(path, index=False, sep=sep, compression=compression)


def _write_stata(path: Path, frame: pd.DataFrame) -> None:
    prepared = _stata_ready(frame)
    prepared.to_stata(path, write_index=False)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def _with_datetime(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        if column in converted.columns:
            converted[column] = pd.to_datetime(converted[column])
    return converted


def _stata_ready(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "class" in prepared.columns:
        prepared = prepared.rename(columns={"class": "opt_class"})
    for column in prepared.columns:
        if prepared[column].dtype == object:
            prepared[column] = prepared[column].fillna("").astype(str)
    return prepared
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype == object:
            prepared[column] = prepared[column].fillna("").astype(str)
    return prepared


def _strike_grid(forward: float, n_strikes: int) -> np.ndarray:
    raw = np.linspace(_STRIKE_MONEYNESS_LO * forward, _STRIKE_MONEYNESS_HI * forward, n_strikes)
    rounded = np.round(raw / _STRIKE_ROUND) * _STRIKE_ROUND
    return np.unique(rounded)


def _as_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _iso(value) -> str:
    return _as_timestamp(value).strftime("%Y-%m-%d")


def _as_expiry_map(expiries_by_date) -> dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]]:
    mapping: dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]] = {}
    for date, groups in expiries_by_date.items():
        mapping[_as_timestamp(date)] = [(_as_timestamp(expiry), int(am_settlement)) for expiry, am_settlement in groups]
    return mapping


def _as_forward_map(
    forwards: Mapping[tuple[object, object, int], float],
) -> dict[tuple[pd.Timestamp, pd.Timestamp, int], float]:
    return {
        (_as_timestamp(date), _as_timestamp(expiry), int(am_settlement)): float(price)
        for (date, expiry, am_settlement), price in forwards.items()
    }


def _as_rate_map(rates) -> dict[pd.Timestamp, float]:
    return {_as_timestamp(date): float(value) for date, value in rates.items()}
