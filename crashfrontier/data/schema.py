from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from crashfrontier.data.parse import (
    parse_cp_flag,
    parse_dates,
    parse_numeric,
    parse_settlement,
)
from crashfrontier.errors import SchemaError

__all__ = [
    "ALIASES",
    "CANONICAL_FIELDS",
    "ColumnMap",
    "REQUIRED_FIELDS",
    "canonicalize",
    "normalize_name",
]

logger = logging.getLogger("crashfrontier")

CANONICAL_FIELDS = (
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
)
REQUIRED_FIELDS = ("date", "expiry", "cp_flag", "strike", "bid", "ask")
OUTPUT_FIELDS = CANONICAL_FIELDS + ("days",)

ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "date",
        "quote_date",
        "quotedate",
        "trade_date",
        "tradedate",
        "obs_date",
        "observation_date",
        "as_of_date",
        "asofdate",
        "datadate",
    ),
    "expiry": (
        "exdate",
        "expiration",
        "expiry",
        "expiration_date",
        "expirationdate",
        "exp_date",
        "expdate",
        "maturity",
        "maturity_date",
    ),
    "cp_flag": (
        "cp_flag",
        "cpflag",
        "call_put",
        "callput",
        "put_call",
        "putcall",
        "option_type",
        "optiontype",
        "type",
        "cp",
        "right",
        "flag",
    ),
    "strike": ("strike_price", "strikeprice", "strike", "strike_px", "k"),
    "bid": ("best_bid", "bestbid", "bid", "bid_price", "bidprice", "bid_px"),
    "ask": ("best_offer", "bestoffer", "best_ask", "bestask", "ask", "offer", "ask_price", "askprice", "ask_px"),
    "volume": ("volume", "vol", "trading_volume", "tradingvolume", "total_volume", "totalvolume"),
    "open_interest": ("open_interest", "openinterest", "oi", "open_int"),
    "impl_volatility": (
        "impl_volatility",
        "implvolatility",
        "implied_volatility",
        "impliedvolatility",
        "impl_vol",
        "implvol",
        "iv",
    ),
    "quote_id": ("optionid", "option_id", "contract_id", "contractid", "id"),
    "am_settlement": ("am_settlement", "amsettlement", "am_set_flag", "amsetflag", "am_flag", "settlement"),
    "symbol": ("symbol", "option_symbol", "optionsymbol", "contract_symbol", "contract", "ticker_symbol"),
    "root": ("root", "root_symbol", "rootsymbol", "underlying", "underlying_symbol"),
    "forward_price": ("forward_price", "forwardprice", "forward", "fwd", "fwd_price"),
    "expiry_indicator": ("expiry_indicator", "expiryindicator"),
}

_SHORT_ALIASES = frozenset({"type", "flag", "id", "k"})
_NUMERIC_FIELDS = ("strike", "bid", "ask", "volume", "open_interest", "impl_volatility", "forward_price")
_STRING_FIELDS = ("symbol", "root", "expiry_indicator")


def normalize_name(name: str) -> str:
    """Strip BOM, quotes, and punctuation so column names can be compared."""
    text = str(name).replace("\ufeff", "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.strip().lower()
    return text.replace(" ", "_").replace("-", "_").replace(".", "_")


@dataclass(frozen=True)
class ColumnMap:
    date: str
    expiry: str
    cp_flag: str
    strike: str
    bid: str
    ask: str
    volume: str | None = None
    open_interest: str | None = None
    impl_volatility: str | None = None
    quote_id: str | None = None
    am_settlement: str | None = None
    symbol: str | None = None
    root: str | None = None
    forward_price: str | None = None
    expiry_indicator: str | None = None

    @classmethod
    def detect(cls, columns: Sequence[str], overrides: dict[str, str] | None = None) -> ColumnMap:
        """Map raw column names onto the canonical option schema."""
        originals = [str(column) for column in columns]
        by_normalized: dict[str, str] = {}
        for original in originals:
            key = normalize_name(original)
            by_normalized.setdefault(key, original)
        claimed: set[str] = set()
        chosen: dict[str, str | None] = {field: None for field in CANONICAL_FIELDS}
        override_map = dict(overrides) if overrides else {}
        unknown = [field for field in override_map if field not in CANONICAL_FIELDS]
        if unknown:
            raise SchemaError(f"unknown override field {unknown[0]!r}")

        for field in CANONICAL_FIELDS:
            match: str | None = None
            if field in override_map:
                candidate = _resolve_override_column(override_map[field], originals, by_normalized)
                if candidate not in claimed:
                    match = candidate
            else:
                match = _first_alias_match(field, by_normalized, claimed)
            if match is not None:
                chosen[field] = match
                claimed.add(match)

        missing_required = [field for field in REQUIRED_FIELDS if chosen[field] is None]
        if missing_required:
            field = missing_required[0]
            raise SchemaError(
                f"missing required field {field!r}; "
                f"aliases {list(ALIASES[field])}; "
                f"columns {originals}"
            )
        return cls(**chosen)

    def usecols(self) -> list[str]:
        """Original column names that are mapped to a canonical field."""
        columns: list[str] = []
        for field in CANONICAL_FIELDS:
            original = getattr(self, field)
            if original is not None:
                columns.append(original)
        return columns

    def rename_map(self) -> dict[str, str]:
        """Map original column names to canonical names."""
        mapping: dict[str, str] = {}
        for field in CANONICAL_FIELDS:
            original = getattr(self, field)
            if original is not None:
                mapping[original] = field
        return mapping

    @classmethod
    def optionmetrics(cls) -> ColumnMap:
        """Column map for standard OptionMetrics names."""
        return cls(
            date="date",
            expiry="exdate",
            cp_flag="cp_flag",
            strike="strike_price",
            bid="best_bid",
            ask="best_offer",
            volume="volume",
            open_interest="open_interest",
            impl_volatility="impl_volatility",
            quote_id="optionid",
            am_settlement="am_settlement",
            symbol="symbol",
            root="root",
            forward_price="forward_price",
            expiry_indicator="expiry_indicator",
        )


def canonicalize(
    raw: pd.DataFrame,
    column_map: ColumnMap,
    *,
    strike_scale: float,
    dayfirst: bool | None,
    cp_values: dict[str, str] | None,
) -> pd.DataFrame:
    """Rename, parse, and coerce a raw option table onto the canonical schema."""
    selected = raw.loc[:, column_map.usecols()].copy()
    selected = selected.rename(columns=column_map.rename_map())

    frame = pd.DataFrame(index=selected.index)
    frame["date"] = parse_dates(selected["date"], dayfirst=dayfirst)
    frame["expiry"] = parse_dates(selected["expiry"], dayfirst=dayfirst)
    for name in _NUMERIC_FIELDS:
        if name in selected.columns:
            frame[name] = parse_numeric(selected[name])
        else:
            frame[name] = float("nan")
    frame["strike"] = frame["strike"] / strike_scale
    frame["cp_flag"] = parse_cp_flag(selected["cp_flag"], cp_values)

    if "quote_id" in selected.columns:
        frame["quote_id"] = _format_quote_id(selected["quote_id"])
        missing_ids = frame["quote_id"].isna()
        if missing_ids.any():
            replacements = [
                f"r{position}"
                for position, is_missing in enumerate(missing_ids.to_numpy())
                if is_missing
            ]
            frame.loc[missing_ids, "quote_id"] = replacements
    else:
        frame["quote_id"] = [f"r{position}" for position in range(len(frame))]

    if "symbol" in selected.columns:
        frame["symbol"] = _as_plain_string(selected["symbol"])
    else:
        frame["symbol"] = ""

    if "symbol" in selected.columns:
        frame["root"] = frame["symbol"].str.split(" ", n=1).str[0]
    elif "root" in selected.columns:
        frame["root"] = _as_plain_string(selected["root"])
    else:
        frame["root"] = ""

    if "expiry_indicator" in selected.columns:
        frame["expiry_indicator"] = _as_plain_string(selected["expiry_indicator"])
    else:
        frame["expiry_indicator"] = ""

    settlement_values = selected["am_settlement"] if "am_settlement" in selected.columns else None
    frame["am_settlement"] = parse_settlement(settlement_values, frame["root"])

    frame["days"] = (frame["expiry"] - frame["date"]).dt.days

    n_before = len(frame)
    frame = frame.dropna(subset=["date", "expiry", "strike", "bid", "ask", "cp_flag"])
    logger.debug("dropped %s rows with missing required fields", n_before - len(frame))

    frame["days"] = frame["days"].astype(int)
    frame["am_settlement"] = frame["am_settlement"].astype(int)
    frame["quote_id"] = frame["quote_id"].astype(str)
    for name in _STRING_FIELDS:
        frame[name] = frame[name].fillna("").astype(str)
    for name in _NUMERIC_FIELDS:
        frame[name] = frame[name].astype("float64")
    return frame.loc[:, list(OUTPUT_FIELDS)].reset_index(drop=True)


def _resolve_override_column(raw_name: str, originals: list[str], by_normalized: dict[str, str]) -> str:
    if raw_name in originals:
        return raw_name
    key = normalize_name(raw_name)
    if key in by_normalized:
        return by_normalized[key]
    raise SchemaError(f"override column {raw_name!r} is not among {originals}")


def _alias_groups(field: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    aliases = ALIASES[field]
    long_names = tuple(alias for alias in aliases if normalize_name(alias) not in _SHORT_ALIASES)
    short_names = tuple(alias for alias in aliases if normalize_name(alias) in _SHORT_ALIASES)
    return long_names, short_names


def _first_alias_match(field: str, by_normalized: dict[str, str], claimed: set[str]) -> str | None:
    long_names, short_names = _alias_groups(field)
    for group in (long_names, short_names):
        for alias in group:
            original = by_normalized.get(normalize_name(alias))
            if original is not None and original not in claimed:
                return original
    return None


def _as_plain_string(values: pd.Series) -> pd.Series:
    missing = values.isna()
    text = values.astype(str)
    text = text.mask(missing, "")
    text = text.mask(text.str.strip().str.lower().isin(("nan", "<na>", "none")), "")
    return text.fillna("").astype(str)


def _format_quote_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    integer_like = numeric.notna() & (numeric == numeric.round())
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    out.loc[integer_like] = numeric.loc[integer_like].round().astype("int64").astype(str)
    leftover = values.notna() & ~integer_like
    out.loc[leftover] = values.loc[leftover].astype(str)
    return out
