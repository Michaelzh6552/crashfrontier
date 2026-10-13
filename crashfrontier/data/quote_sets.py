from __future__ import annotations

import numpy as np
import pandas as pd

from crashfrontier.config import DEFAULT_LM_RANGE

__all__ = [
    "put_universe",
    "otm_put_universe",
    "otm_call_universe",
    "put_pair",
    "local_eight",
    "full_otm",
    "full_panel",
    "nested_quote_sets",
    "select_quote_set",
]

_SORT_KEYS = ("strike", "cp_flag", "quote_id")
_DESIGNS = ("riskneutral", "physical")
_NESTED_NAMES = {
    "riskneutral": ("put_pair", "local_eight", "put_wing"),
    "physical": ("put_pair", "local_eight", "put_wing", "full_otm"),
}
_DEFAULT_LOCAL_SIZE = 8


def put_universe(
    quotes: pd.DataFrame,
    lm_min: float = DEFAULT_LM_RANGE[0],
    lm_max: float = DEFAULT_LM_RANGE[1],
) -> pd.DataFrame:
    """Puts whose log-moneyness lies in ``[lm_min, lm_max]``."""
    log_m = quotes["log_moneyness"]
    mask = quotes["cp_flag"].eq("P") & log_m.ge(lm_min) & log_m.le(lm_max)
    return _finish(quotes.loc[mask])


def otm_put_universe(
    quotes: pd.DataFrame, lm_min: float = DEFAULT_LM_RANGE[0]
) -> pd.DataFrame:
    """Puts with ``x <= 1`` and log-moneyness at least ``lm_min``."""
    mask = (
        quotes["cp_flag"].eq("P")
        & quotes["x"].le(1.0)
        & quotes["log_moneyness"].ge(lm_min)
    )
    return _finish(quotes.loc[mask])


def otm_call_universe(quotes: pd.DataFrame) -> pd.DataFrame:
    """Calls with ``x >= 1``."""
    mask = quotes["cp_flag"].eq("C") & quotes["x"].ge(1.0)
    return _finish(quotes.loc[mask])


def put_pair(puts: pd.DataFrame, K) -> pd.DataFrame:
    """Puts at the two strikes that bracket ``K``.

    1. ``below`` is the largest ``x <= K``; ``above`` is the smallest ``x >= K``.
    2. If those strikes differ, both are kept.
    3. If they coincide (a quote sits at ``K``), keep that strike and the
       nearest remaining unique strike.
    4. If two distinct bracketing strikes are unavailable, keep the two unique
       strikes closest to ``K``.
    5. Every put row at the chosen strikes is returned.
    """
    panel = _puts_only(puts)
    if panel.empty:
        return _finish(panel)
    threshold = float(K)
    unique_x = _unique_x(panel)
    below = unique_x[unique_x <= threshold]
    above = unique_x[unique_x >= threshold]
    if below.size and above.size:
        lo = float(below[-1])
        hi = float(above[0])
        if lo != hi:
            chosen = (lo, hi)
        else:
            others = unique_x[unique_x != lo]
            if others.size == 0:
                chosen = (lo,)
            else:
                nearest = others[np.argmin(np.abs(others - threshold))]
                chosen = (lo, float(nearest))
    else:
        distances = np.abs(unique_x - threshold)
        order = np.lexsort((unique_x, distances))
        chosen = tuple(unique_x[order[: min(2, unique_x.size)]])
    return _rows_at_x(panel, chosen)


def local_eight(puts: pd.DataFrame, K, size: int = _DEFAULT_LOCAL_SIZE) -> pd.DataFrame:
    """``put_pair`` strikes plus the nearest remaining unique strikes, up to ``size``."""
    panel = _puts_only(puts)
    pair = put_pair(panel, K)
    return _extend_pair(panel, pair, K, size)


def _extend_pair(
    panel: pd.DataFrame, pair: pd.DataFrame, K: float, size: int
) -> pd.DataFrame:
    pair_x = _unique_x(pair)
    extra = max(0, int(size) - int(pair_x.size))
    remaining = _unique_x(panel)
    if pair_x.size:
        remaining = remaining[~np.isin(remaining, pair_x)]
    if extra == 0 or remaining.size == 0:
        chosen = pair_x
    else:
        threshold = float(K)
        distances = np.abs(remaining - threshold)
        order = np.lexsort((remaining, distances))
        chosen = np.concatenate([pair_x, remaining[order[:extra]]])
    return _rows_at_x(panel, chosen)


def full_otm(
    quotes: pd.DataFrame, lm_min: float = DEFAULT_LM_RANGE[0]
) -> pd.DataFrame:
    """Union of the OTM put universe and the OTM call universe."""
    puts = otm_put_universe(quotes, lm_min)
    calls = otm_call_universe(quotes)
    if puts.empty:
        return calls
    if calls.empty:
        return puts
    return _finish(pd.concat([puts, calls], ignore_index=True))


def full_panel(quotes: pd.DataFrame) -> pd.DataFrame:
    """All call and put rows."""
    return _finish(quotes.loc[quotes["cp_flag"].isin(("C", "P"))])


def nested_quote_sets(
    quotes: pd.DataFrame, K, design: str = "riskneutral"
) -> dict[str, pd.DataFrame]:
    """Nested quote collections for a risk-neutral or physical design."""
    names = _nested_names(design)
    if design == "riskneutral":
        puts = put_universe(quotes)
        sets = {
            "put_pair": put_pair(puts, K),
            "local_eight": local_eight(puts, K),
            "put_wing": puts,
        }
    else:
        puts = otm_put_universe(quotes)
        pair = _physical_put_pair(puts, K)
        sets = {
            "put_pair": pair,
            "local_eight": _extend_pair(puts, pair, K, _DEFAULT_LOCAL_SIZE),
            "put_wing": puts,
            "full_otm": full_otm(quotes),
        }
    return {name: sets[name] for name in names}


def select_quote_set(quotes: pd.DataFrame, K, name, design) -> pd.DataFrame:
    """One named collection from ``nested_quote_sets``, or ``full_panel``."""
    available = _available_names(design)
    if name not in available:
        raise ValueError(
            f"unknown quote set {name!r}; available names: {', '.join(available)}"
        )
    if name == "full_panel":
        return full_panel(quotes)
    if name == "full_otm":
        return full_otm(quotes)
    if name == "put_wing":
        return put_universe(quotes) if design == "riskneutral" else otm_put_universe(quotes)
    return nested_quote_sets(quotes, K, design)[name]


def _nested_names(design: str) -> tuple[str, ...]:
    names = _NESTED_NAMES.get(design)
    if names is None:
        raise ValueError(
            f"unknown design {design!r}; available names: {', '.join(_DESIGNS)}"
        )
    return names


def _available_names(design: str) -> tuple[str, ...]:
    return _nested_names(design) + ("full_panel",)


def _puts_only(quotes: pd.DataFrame) -> pd.DataFrame:
    return quotes.loc[quotes["cp_flag"].eq("P")]


def _physical_put_pair(puts: pd.DataFrame, K: float) -> pd.DataFrame:
    panel = _puts_only(puts)
    threshold = float(K)
    unique_x = _unique_x(panel)
    below = unique_x[unique_x < threshold]
    above = unique_x[unique_x > threshold]
    if below.size == 0 or above.size == 0:
        raise ValueError("physical put_pair requires distinct put strikes bracketing K")
    return _rows_at_x(panel, (float(below[-1]), float(above[0])))


def _unique_x(quotes: pd.DataFrame) -> np.ndarray:
    if quotes.empty:
        return np.array([], dtype=float)
    return np.unique(quotes["x"].to_numpy(dtype=float))


def _rows_at_x(quotes: pd.DataFrame, values) -> pd.DataFrame:
    chosen = np.asarray(values, dtype=float)
    if quotes.empty or chosen.size == 0:
        return _finish(quotes.iloc[0:0])
    mask = np.isin(quotes["x"].to_numpy(dtype=float), chosen)
    return _finish(quotes.loc[mask])


def _finish(quotes: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in _SORT_KEYS if key in quotes.columns]
    if keys:
        ordered = quotes.sort_values(keys, kind="mergesort")
    else:
        ordered = quotes
    return ordered.reset_index(drop=True)
