from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crashfrontier.errors import SchemaError

__all__ = [
    "DATE_FORMATS",
    "detect_date_format",
    "detect_strike_scale",
    "parse_cp_flag",
    "parse_dates",
    "parse_numeric",
    "parse_settlement",
]

logger = logging.getLogger("crashfrontier")

DATE_FORMATS: tuple[str, ...] = (
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

_YYYYMMDD_MIN = 19000101
_YYYYMMDD_MAX = 21001231
_SAS_DAY_MIN = 5000
_SAS_EXCEL_BOUNDARY = 30000
_EXCEL_DAY_MAX = 80000
_DATE_SAMPLE_SIZE = 200
_STRIKE_SCALE_CANDIDATES = (1.0, 10.0, 100.0, 1000.0)
_STRIKE_UNREFERENCED_CUTOFF = 100000.0

_AMBIGUOUS_PAIRS = (
    ("%m/%d/%Y", "%d/%m/%Y", "/"),
    ("%m-%d-%Y", "%d-%m-%Y", "-"),
)

_SETTLEMENT_ONE = frozenset({"AM", "A", "1", "TRUE", "Y"})
_SETTLEMENT_ZERO = frozenset({"PM", "P", "0", "FALSE", "N"})


def parse_dates(values: pd.Series, *, dayfirst: bool | None = None) -> pd.Series:
    """Parse observation or expiry dates to midnight datetime64[ns]."""
    series = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return _normalize_datetime(series)
    if _is_numeric_or_digit_strings(series):
        return _parse_numeric_dates(series)
    fmt = _match_string_format(series, dayfirst=dayfirst)
    if fmt is not None:
        return _normalize_datetime(_parse_with_format(series, fmt))
    logger.warning("no explicit date format matched; using pandas to_datetime")
    parsed = pd.to_datetime(series, errors="raise", dayfirst=bool(dayfirst))
    return _normalize_datetime(parsed)


def detect_date_format(sample: pd.Series, *, dayfirst: bool | None = None) -> str | None:
    """Return the date format token that would be used, or None."""
    series = pd.Series(sample)
    if pd.api.types.is_datetime64_any_dtype(series):
        return None
    if _is_numeric_or_digit_strings(series):
        numeric = pd.to_numeric(series, errors="coerce")
        kind = _numeric_date_kind(numeric.dropna())
        if kind is None:
            return None
        return {"yyyymmdd": "%Y%m%d", "sas": "sas_days", "excel": "excel_serial"}[kind]
    return _match_string_format(series, dayfirst=dayfirst)


def parse_numeric(values: pd.Series) -> pd.Series:
    """Parse a numeric column, stripping thousands separators and dollar signs."""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    text = text.mask(text.eq("") | text.eq(".") | text.str.lower().eq("nan"), pd.NA)
    return pd.to_numeric(text, errors="coerce")


def parse_cp_flag(values: pd.Series, cp_values: dict[str, str] | None = None) -> pd.Series:
    """Parse call/put flags to ``C`` or ``P``."""
    series = pd.Series(values)
    missing = _is_missing_text(series)
    text = series.astype(str).str.strip().str.upper()
    mapped = pd.Series(pd.NA, index=series.index, dtype="object")
    if cp_values:
        table = {str(key).strip().upper(): _canonical_cp(value) for key, value in cp_values.items()}
        mapped = text.map(table)
    first = text.str[:1]
    from_letter = pd.Series(pd.NA, index=series.index, dtype="object")
    from_letter = from_letter.mask(first.eq("C"), "C")
    from_letter = from_letter.mask(first.eq("P"), "P")
    result = mapped.fillna(from_letter)
    result = result.mask(missing, pd.NA)
    unresolved = result.isna() & ~missing
    if unresolved.any():
        distinct = sorted({str(item) for item in text[unresolved].unique().tolist()})
        raise SchemaError(f"unrecognized cp_flag values: {distinct}")
    return result


def parse_settlement(values: pd.Series | None, root: pd.Series) -> pd.Series:
    """Parse AM/PM settlement flags to 0/1 integers."""
    roots = pd.Series(root)
    inferred = (roots.astype(str) == "SPX").astype(int)
    if values is None:
        return inferred.astype(int)
    series = pd.Series(values)
    if len(series) != len(roots):
        raise SchemaError("settlement values and root must have the same length")
    series.index = roots.index
    parsed = pd.to_numeric(_parse_settlement_values(series), errors="coerce")
    result = inferred.astype(int).copy()
    known = parsed.notna()
    result.loc[known] = parsed.loc[known].astype(int)
    return result.astype(int)


def detect_strike_scale(strikes: pd.Series, reference_price: float) -> float:
    """Infer whether strikes are stored as 1, 10, 100, or 1000 times the index unit."""
    median_strike = float(pd.to_numeric(strikes, errors="coerce").median())
    if (
        np.isfinite(reference_price)
        and reference_price > 0.0
        and np.isfinite(median_strike)
        and median_strike > 0.0
    ):
        scores = [
            abs(float(np.log((median_strike / candidate) / reference_price)))
            for candidate in _STRIKE_SCALE_CANDIDATES
        ]
        return float(_STRIKE_SCALE_CANDIDATES[int(np.argmin(np.asarray(scores)))])
    scale = 1000.0 if np.isfinite(median_strike) and median_strike > _STRIKE_UNREFERENCED_CUTOFF else 1.0
    logger.warning(
        "no finite reference price for strike scale; inferred %s from median strike; pass strike_scale explicitly",
        scale,
    )
    return scale


def _canonical_cp(value: object) -> str:
    token = str(value).strip().upper()
    if token[:1] in {"C", "P"}:
        return token[:1]
    return token


def _is_missing_text(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series.isna()
    text = series.astype(str).str.strip()
    return series.isna() | text.eq("") | text.str.lower().isin(("nan", "<na>", "none", "nat"))


def _normalize_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values)
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _is_numeric_or_digit_strings(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return True
    text = values.dropna().astype(str).str.strip()
    if text.empty:
        return False
    text = text[~text.str.lower().isin(("nan", "<na>", "none", "nat")) & text.ne("")]
    if text.empty:
        return False
    return bool(text.str.fullmatch(r"\d+(?:\.0+)?").all())


def _numeric_date_kind(non_null: pd.Series) -> str | None:
    if non_null.empty:
        return None
    integers = non_null.astype("int64").to_numpy()
    if np.all((integers >= _YYYYMMDD_MIN) & (integers <= _YYYYMMDD_MAX)):
        return "yyyymmdd"
    if np.all((integers >= _SAS_DAY_MIN) & (integers <= _SAS_EXCEL_BOUNDARY)):
        return "sas"
    if np.all((integers >= _SAS_EXCEL_BOUNDARY) & (integers <= _EXCEL_DAY_MAX)):
        return "excel"
    distinct = sorted({int(value) for value in integers[:8].tolist()})
    raise SchemaError(f"unrecognized numeric date values: {distinct}")


def _parse_numeric_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    kind = _numeric_date_kind(numeric.dropna())
    if kind is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if kind == "yyyymmdd":
        logger.warning("interpreting numeric dates as YYYYMMDD")
        valid = numeric.notna()
        converted = pd.to_datetime(numeric.loc[valid].astype("int64").astype(str), format="%Y%m%d")
        return _normalize_datetime(converted.reindex(values.index))
    if kind == "sas":
        logger.warning("interpreting numeric dates as SAS days from 1960-01-01")
        return _normalize_datetime(pd.to_datetime(numeric, unit="D", origin="1960-01-01"))
    logger.warning("interpreting numeric dates as Excel serial numbers")
    return _normalize_datetime(pd.to_datetime(numeric, unit="D", origin="1899-12-30"))


def _unique_sample(values: pd.Series) -> pd.Series:
    text = values.dropna().astype(str).str.strip()
    text = text[text.ne("") & ~text.str.lower().isin(("nan", "<na>", "none", "nat"))]
    unique = pd.unique(text.to_numpy())[:_DATE_SAMPLE_SIZE]
    return pd.Series(unique, dtype="object")


def _trial_values(sample: pd.Series, fmt: str) -> pd.Series:
    if "%b" in fmt:
        return sample.astype(str).str.title()
    return sample.astype(str)


def _format_accepted(sample: pd.Series, fmt: str) -> bool:
    parsed = pd.to_datetime(_trial_values(sample, fmt), format=fmt, errors="coerce")
    return bool(parsed.notna().all())


def _day_month_order(sample: pd.Series, *, dayfirst: bool | None, separator: str) -> str:
    if dayfirst is True:
        return "dmy"
    if dayfirst is False:
        return "mdy"
    first_gt_twelve = False
    second_gt_twelve = False
    for raw in sample.astype(str):
        parts = raw.split(separator)
        if len(parts) < 2:
            continue
        try:
            first = int(parts[0])
            second = int(parts[1])
        except ValueError:
            continue
        if first > 12:
            first_gt_twelve = True
        if second > 12:
            second_gt_twelve = True
    if first_gt_twelve:
        return "dmy"
    if second_gt_twelve:
        return "mdy"
    logger.warning("ambiguous day/month date format; assuming month-first")
    return "mdy"


def _match_string_format(values: pd.Series, *, dayfirst: bool | None) -> str | None:
    sample = _unique_sample(values)
    if sample.empty:
        return None
    accepted = [fmt for fmt in DATE_FORMATS if _format_accepted(sample, fmt)]
    accepted_set = set(accepted)
    for month_first, day_first, separator in _AMBIGUOUS_PAIRS:
        if month_first in accepted_set and day_first in accepted_set:
            order = _day_month_order(sample, dayfirst=dayfirst, separator=separator)
            drop = month_first if order == "dmy" else day_first
            accepted_set.remove(drop)
    ordered = [fmt for fmt in DATE_FORMATS if fmt in accepted_set]
    if not ordered:
        return None
    return ordered[0]


def _parse_with_format(values: pd.Series, fmt: str) -> pd.Series:
    missing = _is_missing_text(values)
    present = values.loc[~missing]
    converted = pd.to_datetime(_trial_values(present, fmt), format=fmt, errors="raise")
    return converted.reindex(values.index)


def _parse_settlement_values(values: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(values, errors="coerce")
    present = ~_is_missing_text(values)
    use_numeric = pd.api.types.is_numeric_dtype(values)
    if not use_numeric and present.any():
        use_numeric = bool(coerced.loc[present].notna().all())
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    if use_numeric:
        out.loc[coerced.notna()] = (coerced.loc[coerced.notna()] != 0).astype(int)
        return out
    text = values.astype(str).str.strip().str.upper()
    mapped = pd.Series(pd.NA, index=values.index, dtype="object")
    mapped = mapped.mask(text.isin(_SETTLEMENT_ONE), 1)
    mapped = mapped.mask(text.isin(_SETTLEMENT_ZERO), 0)
    mapped = mapped.mask(~present, pd.NA)
    unresolved = mapped.isna() & present
    if unresolved.any():
        distinct = sorted({str(item) for item in text[unresolved].unique().tolist()})
        raise SchemaError(f"unrecognized settlement values: {distinct}")
    return mapped
