from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from crashfrontier.config import (
    DEFAULT_SERIES_TARGET,
    DEFAULT_SERIES_WINDOW,
    DEFAULT_TENORS,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_FRACTION,
    SERIES_CACHE_FLUSH,
    WEEKDAY_PRIORITY,
)
from crashfrontier.data.cross_section import CrossSection
from crashfrontier.data.dataset import Dataset
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.parse import parse_dates
from crashfrontier.data.quote_sets import select_quote_set
from crashfrontier.errors import CrashFrontierError, NoSuchDateError
from crashfrontier.frontier import Frontier, compute_frontier

__all__ = [
    "frontier_series",
    "pick_dates",
    "term_structure",
    "threshold_profile",
]

logger = logging.getLogger("crashfrontier")

_CACHE_SIGNATURE = "cache_signature"

_SUMMARY_COLUMNS = (
    "date",
    "expiry",
    "days",
    "am_settlement",
    "K",
    "gamma",
    "support_cap",
    "quote_set",
    "n_quotes",
    "n_strikes",
    "forward",
    "forward_source",
    "rate",
    "discount",
    "spot",
    "mode",
    "method",
    "status",
    "dimension",
    "prob_lo",
    "prob_hi",
    "loss_lo",
    "loss_hi",
    "depth_lo",
    "depth_hi",
    "area",
    "area_outer",
    "baseline_area",
    "area_ratio",
    "interior_ok",
    "interior_margin",
    "support_calls",
    "max_gap",
    "solver_gap",
    "elapsed_seconds",
)
_ITEM_ERRORS = (CrashFrontierError, ValueError)
_WORKER_DATASET: Dataset | None = None


def term_structure(
    ds: Dataset,
    date,
    *,
    tenors: Sequence[int] = DEFAULT_TENORS,
    window_fraction: float = DEFAULT_WINDOW_FRACTION,
    K: float = DEFAULT_THRESHOLD,
    gamma: int = 0,
    am_settlement: int | None = None,
    **frontier_kwargs,
) -> tuple[pd.DataFrame, dict[int, Frontier]]:
    """Crash frontiers for several target maturities on one observation date."""
    stamp = _as_timestamp(date)
    min_days, max_days = _day_limits(ds)
    fraction = float(window_fraction)
    selected: dict[int, CrossSection] = {}
    errors: dict[int, str] = {}
    sections: dict[int, CrossSection] = {}
    for tenor in (int(item) for item in tenors):
        low = max(min_days, int(round(tenor * (1.0 - fraction))))
        high = min(max_days, int(round(tenor * (1.0 + fraction))))
        try:
            section = ds.cross_section(
                stamp,
                target_days=tenor,
                window=(low, high),
                am_settlement=am_settlement,
            )
        except _ITEM_ERRORS as exc:
            logger.warning("term_structure target_days=%s: %s", tenor, exc)
            errors[tenor] = str(exc)
            continue
        sections[tenor] = section
        selected[tenor] = section
    _drop_duplicate_expiries(selected, errors)
    fronts: dict[int, Frontier] = {}
    rows: list[dict[str, object]] = []
    for tenor in (int(item) for item in tenors):
        if tenor in errors:
            section = sections.get(tenor)
            rows.append(
                _error_row(
                    errors[tenor],
                    K=K,
                    gamma=gamma,
                    section=section,
                    target_days=tenor,
                    quote_set=frontier_kwargs.get("quote_set"),
                    design=frontier_kwargs.get("design"),
                )
            )
            continue
        section = selected[tenor]
        try:
            frontier = compute_frontier(
                section, float(K), gamma=int(gamma), **frontier_kwargs
            )
        except _ITEM_ERRORS as exc:
            logger.warning("term_structure target_days=%s: %s", tenor, exc)
            rows.append(
                _error_row(
                    str(exc),
                    K=K,
                    gamma=gamma,
                    section=section,
                    target_days=tenor,
                    quote_set=frontier_kwargs.get("quote_set"),
                    design=frontier_kwargs.get("design"),
                    exc=exc,
                )
            )
            continue
        fronts[tenor] = frontier
        rows.append(_row_from_frontier(frontier, target_days=int(tenor), error=""))
    return _finalize_table(rows, ("target_days", "error")), fronts


def threshold_profile(
    cs: CrossSection,
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
    *,
    gamma: int = 0,
    **frontier_kwargs,
) -> tuple[pd.DataFrame, dict[float, Frontier]]:
    """Crash frontiers of one cross-section across several thresholds."""
    fronts: dict[float, Frontier] = {}
    rows: list[dict[str, object]] = []
    for raw in thresholds:
        threshold = float(raw)
        try:
            frontier = compute_frontier(
                cs, threshold, gamma=int(gamma), **frontier_kwargs
            )
        except _ITEM_ERRORS as exc:
            logger.warning("threshold_profile K=%s: %s", threshold, exc)
            rows.append(
                _error_row(
                    str(exc),
                    K=threshold,
                    gamma=gamma,
                    section=cs,
                    quote_set=frontier_kwargs.get("quote_set"),
                    design=frontier_kwargs.get("design"),
                    exc=exc,
                )
            )
            continue
        fronts[threshold] = frontier
        rows.append(_row_from_frontier(frontier, error=""))
    return _finalize_table(rows, ("error",)), fronts


def pick_dates(
    available: Sequence[pd.Timestamp],
    *,
    start=None,
    end=None,
    freq: str = "W",
    weekday_priority: Sequence[int] = WEEKDAY_PRIORITY,
) -> list[pd.Timestamp]:
    """Subsample observation dates at daily, weekly, monthly, or quarterly frequency."""
    dates = sorted({_as_timestamp(item) for item in available})
    if start is not None:
        start_stamp = _as_timestamp(start)
        dates = [item for item in dates if item >= start_stamp]
    if end is not None:
        end_stamp = _as_timestamp(end)
        dates = [item for item in dates if item <= end_stamp]
    if not dates:
        return []
    key = str(freq).upper()
    if key == "D":
        return dates
    if key == "W":
        return _pick_weekly(dates, weekday_priority)
    if key == "M":
        return _last_in_group(dates, lambda stamp: (stamp.year, stamp.month))
    if key == "Q":
        return _last_in_group(dates, lambda stamp: (stamp.year, stamp.quarter))
    raise ValueError(f"freq must be D, W, M, or Q, got {freq!r}")


def frontier_series(
    ds: Dataset,
    dates: Sequence | None = None,
    *,
    start=None,
    end=None,
    freq: str = "W",
    K: float = DEFAULT_THRESHOLD,
    gamma: int = 0,
    quote_set=None,
    days: int | None = None,
    target_days: int | None = DEFAULT_SERIES_TARGET,
    window: tuple[int, int] | None = DEFAULT_SERIES_WINDOW,
    am_settlement: int | None = None,
    snap: str | None = "backward",
    workers: int = 1,
    cache_path: str | Path | None = None,
    on_error: str = "skip",
    **frontier_kwargs,
) -> pd.DataFrame:
    """Crash-frontier summaries along a date list or a sampled date range."""
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    if snap not in {"backward", "forward", None}:
        raise ValueError("snap must be 'backward', 'forward', or None")
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    if dates is None:
        requested = pick_dates(ds.available_dates(), start=start, end=end, freq=freq)
    else:
        requested = [_as_timestamp(item) for item in dates]
    extra = ("requested_date", "error")
    if not requested:
        return _finalize_table([], extra)

    resolved_days = None if days is None else int(days)
    resolved_target = None if days is not None or target_days is None else int(target_days)
    resolved_window = (
        None
        if days is not None or window is None
        else (int(window[0]), int(window[1]))
    )
    signature = _series_signature(
        ds,
        K=float(K),
        gamma=int(gamma),
        quote_set=quote_set,
        days=resolved_days,
        target_days=resolved_target,
        window=resolved_window,
        am_settlement=am_settlement,
        snap=snap,
        frontier_kwargs=dict(frontier_kwargs),
    )
    cache_file = None if cache_path is None else Path(cache_path)
    cached_map = _load_cache_map(cache_file, signature)
    pending = [stamp for stamp in requested if stamp not in cached_map]
    computed: list[dict[str, object]] = []
    if pending:
        spec, n_workers = _series_spec(ds, n_workers)
        job_kwargs = {
            "K": float(K),
            "gamma": int(gamma),
            "quote_set": quote_set,
            "days": resolved_days,
            "target_days": resolved_target,
            "window": resolved_window,
            "am_settlement": am_settlement,
            "snap": snap,
            "on_error": on_error,
            "frontier_kwargs": dict(frontier_kwargs),
        }
        if n_workers == 1:
            computed = _run_series_sequential(
                ds, pending, job_kwargs, cache_file, list(cached_map.values()), signature
            )
        else:
            computed = _run_series_parallel(
                spec,
                pending,
                job_kwargs,
                n_workers,
                cache_file,
                list(cached_map.values()),
                signature,
            )
    by_requested = dict(cached_map)
    for row in computed:
        by_requested[_as_timestamp(row["requested_date"])] = row
    ordered = [by_requested[stamp] for stamp in requested]
    table = _finalize_table(ordered, extra)
    if cache_file is not None:
        _write_series_csv(cache_file, table, signature)
    return table


def _init_series_worker(spec: dict) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = Dataset.from_spec(spec)


def _series_worker(job: dict) -> tuple[int, dict[str, object]]:
    dataset = _WORKER_DATASET
    if dataset is None:
        dataset = Dataset.from_spec(job["spec"])
    requested = _as_timestamp(job["requested"])
    row = _evaluate_series_date(
        dataset,
        requested,
        K=job["K"],
        gamma=job["gamma"],
        quote_set=job["quote_set"],
        days=job["days"],
        target_days=job["target_days"],
        window=job["window"],
        am_settlement=job["am_settlement"],
        snap=job["snap"],
        on_error=job["on_error"],
        frontier_kwargs=job["frontier_kwargs"],
    )
    return int(job["index"]), row


def _run_series_sequential(
    ds: Dataset,
    pending: list[pd.Timestamp],
    job_kwargs: dict,
    cache_file: Path | None,
    cached_rows: list[dict[str, object]],
    signature: str,
) -> list[dict[str, object]]:
    computed: list[dict[str, object]] = []
    for stamp in pending:
        computed.append(_evaluate_series_date(ds, stamp, **job_kwargs))
        if cache_file is not None and len(computed) % SERIES_CACHE_FLUSH == 0:
            _write_series_csv(
                cache_file,
                _finalize_table(cached_rows + computed, ("requested_date", "error")),
                signature,
            )
    return computed


def _run_series_parallel(
    spec: dict,
    pending: list[pd.Timestamp],
    job_kwargs: dict,
    n_workers: int,
    cache_file: Path | None,
    cached_rows: list[dict[str, object]],
    signature: str,
) -> list[dict[str, object]]:
    jobs = [_series_job(index, stamp, spec, job_kwargs) for index, stamp in enumerate(pending)]
    by_index: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_series_worker,
        initargs=(spec,),
    ) as pool:
        futures = [pool.submit(_series_worker, job) for job in jobs]
        progress = tqdm(as_completed(futures), total=len(futures), desc="frontier_series")
        finished = 0
        for future in progress:
            index, row = future.result()
            by_index[int(index)] = row
            finished += 1
            if cache_file is not None and finished % SERIES_CACHE_FLUSH == 0:
                _write_series_csv(
                    cache_file,
                    _finalize_table(
                        cached_rows + [by_index[i] for i in sorted(by_index)],
                        ("requested_date", "error"),
                    ),
                    signature,
                )
    return [by_index[index] for index in range(len(pending))]


def _series_job(index: int, stamp: pd.Timestamp, spec: dict, job_kwargs: dict) -> dict:
    job = dict(job_kwargs)
    job["index"] = int(index)
    job["requested"] = pd.Timestamp(stamp).strftime("%Y-%m-%d")
    job["spec"] = spec
    return job


def _series_spec(ds: Dataset, n_workers: int) -> tuple[dict | None, int]:
    if n_workers <= 1:
        return None, 1
    try:
        return ds.spec(), n_workers
    except ValueError as exc:
        logger.warning("%s; using workers=1", exc)
        return None, 1


def _evaluate_series_date(
    ds: Dataset,
    requested: pd.Timestamp,
    *,
    K: float,
    gamma: int,
    quote_set,
    days: int | None,
    target_days: int | None,
    window: tuple[int, int] | None,
    am_settlement: int | None,
    snap: str | None,
    on_error: str,
    frontier_kwargs: dict,
) -> dict[str, object]:
    section: CrossSection | None = None
    try:
        used = _snap_observation(ds, requested, snap)
        section = ds.cross_section(
            used,
            days=days,
            target_days=target_days,
            window=window,
            am_settlement=am_settlement,
        )
        frontier = compute_frontier(
            section,
            float(K),
            gamma=int(gamma),
            quote_set=quote_set,
            **frontier_kwargs,
        )
        return _row_from_frontier(frontier, requested_date=requested, error="")
    except _ITEM_ERRORS as exc:
        if on_error == "raise":
            raise
        logger.warning("frontier_series %s: %s", requested.date(), exc)
        return _error_row(
            str(exc),
            K=K,
            gamma=gamma,
            section=section,
            requested_date=requested,
            quote_set=quote_set,
            design=frontier_kwargs.get("design"),
            exc=exc,
        )


def _snap_observation(ds: Dataset, requested: pd.Timestamp, snap: str | None) -> pd.Timestamp:
    available = set(ds.available_dates())
    if requested in available:
        return requested
    if snap is None:
        raise NoSuchDateError(f"{requested.date()} is not an option date")
    return ds.nearest_date(requested, direction=snap)


def _drop_duplicate_expiries(
    selected: dict[int, CrossSection],
    errors: dict[int, str],
) -> None:
    by_expiry: dict[pd.Timestamp, list[int]] = {}
    for tenor, section in selected.items():
        expiry = pd.Timestamp(section.expiry).normalize()
        by_expiry.setdefault(expiry, []).append(int(tenor))
    for _expiry, tenors in by_expiry.items():
        if len(tenors) < 2:
            continue
        ranked = sorted(
            tenors,
            key=lambda tenor: (abs(int(selected[tenor].days) - tenor), tenors.index(tenor)),
        )
        keep = ranked[0]
        for tenor in ranked[1:]:
            errors[tenor] = "expiry already used"
            selected.pop(tenor, None)
            logger.warning(
                "term_structure target_days=%s: expiry already used; kept target_days=%s",
                tenor,
                keep,
            )


def _pick_weekly(dates: list[pd.Timestamp], weekday_priority: Sequence[int]) -> list[pd.Timestamp]:
    groups: dict[object, list[pd.Timestamp]] = {}
    for stamp in dates:
        groups.setdefault(stamp.to_period("W-SUN"), []).append(stamp)
    priority = [int(item) for item in weekday_priority]
    chosen: list[pd.Timestamp] = []
    for period in sorted(groups):
        members = groups[period]
        by_weekday = {int(item.weekday()): item for item in members}
        pick = None
        for weekday in priority:
            if weekday in by_weekday:
                pick = by_weekday[weekday]
                break
        if pick is None:
            pick = members[-1]
        chosen.append(pick)
    return chosen


def _last_in_group(dates: list[pd.Timestamp], key) -> list[pd.Timestamp]:
    groups: dict[object, list[pd.Timestamp]] = {}
    for stamp in dates:
        groups.setdefault(key(stamp), []).append(stamp)
    return [groups[item][-1] for item in sorted(groups)]


def _day_limits(ds: Dataset) -> tuple[int, int]:
    filt = getattr(ds, "_quote_filter", QuoteFilter())
    return int(filt.min_days), int(filt.max_days)


def _row_from_frontier(frontier: Frontier, **extra) -> dict[str, object]:
    row = frontier.summary()
    row.update(extra)
    return row


def _error_row(
    message: str,
    *,
    K: float,
    gamma: int,
    section: CrossSection | None = None,
    quote_set=None,
    design: str | None = None,
    exc: BaseException | None = None,
    **extra,
) -> dict[str, object]:
    row: dict[str, object] = {name: float("nan") for name in _SUMMARY_COLUMNS}
    row["date"] = pd.NaT
    row["expiry"] = pd.NaT
    row["K"] = float(K)
    row["gamma"] = int(gamma)
    row["quote_set"] = None
    row["forward_source"] = None
    row["mode"] = None
    row["method"] = None
    row["status"] = None
    row["interior_ok"] = None
    row["error"] = str(message)
    if section is not None:
        row["date"] = section.date
        row["expiry"] = section.expiry
        row["days"] = int(section.days)
        row["am_settlement"] = int(section.am_settlement)
        row["forward"] = float(section.forward)
        row["forward_source"] = str(section.forward_source)
        row["rate"] = float(section.rate)
        row["discount"] = float(section.discount)
        row["spot"] = float(section.spot)
        row["n_quotes"] = int(len(section.quotes))
    row.update(extra)
    _keep_error_quote_fields(
        row,
        K=float(K),
        gamma=int(gamma),
        section=section,
        quote_set=quote_set,
        design=design,
        exc=exc,
    )
    row["error"] = str(message)
    return row


def _keep_error_quote_fields(
    row: dict[str, object],
    *,
    K: float,
    gamma: int,
    section: CrossSection | None,
    quote_set,
    design: str | None,
    exc: BaseException | None,
) -> None:
    frame = quote_set if isinstance(quote_set, pd.DataFrame) else None
    name = "custom" if frame is not None else (None if quote_set is None else str(quote_set))
    tagged = getattr(exc, "quote_set", None) if exc is not None else None
    if tagged:
        name = str(tagged)
    if name is not None:
        row["quote_set"] = name
    counted = _error_strike_count(
        section, frame, name, K=K, gamma=gamma, design=design
    )
    if counted is not None:
        row["n_strikes"] = counted


def _error_strike_count(
    section: CrossSection | None,
    frame: pd.DataFrame | None,
    name: str | None,
    *,
    K: float,
    gamma: int,
    design: str | None,
) -> int | None:
    if frame is not None:
        counted = _strike_count(frame)
        if counted is not None:
            return counted
    if section is not None and name:
        try:
            selected = select_quote_set(
                section.quotes, float(K), name, _design_name(gamma, design)
            )
        except (CrashFrontierError, KeyError, TypeError, ValueError):
            selected = None
        if selected is not None:
            counted = _strike_count(selected)
            if counted is not None:
                return counted
    if section is None:
        return None
    return _strike_count(section.quotes)


def _strike_count(frame: pd.DataFrame) -> int | None:
    if "strike" not in frame.columns:
        return None
    return int(frame["strike"].nunique()) if len(frame) else 0


def _design_name(gamma: int, design: str | None) -> str:
    if design is not None:
        return str(design)
    return "riskneutral" if int(gamma) == 0 else "physical"


def _finalize_table(rows: list[dict[str, object]], extra: Sequence[str]) -> pd.DataFrame:
    columns = list(_SUMMARY_COLUMNS)
    for name in extra:
        if name not in columns:
            columns.append(name)
    frame = pd.DataFrame.from_records(list(rows), columns=columns)
    for name in ("date", "expiry", "requested_date"):
        if name in frame.columns:
            frame[name] = _naive_datetimes(frame[name])
    if "error" in frame.columns:
        frame["error"] = frame["error"].fillna("").astype(str)
        frame.loc[frame["error"].str.lower().eq("nan"), "error"] = ""
    for name in ("support_cap", "quote_set", "forward_source", "mode", "method", "status"):
        if name in frame.columns:
            frame[name] = frame[name].where(pd.notna(frame[name]), np.nan)
    if "requested_date" in frame.columns and not frame.empty:
        frame = frame.sort_values("requested_date", kind="mergesort").reset_index(drop=True)
    return frame


def _load_cache_map(path: Path | None, signature: str) -> dict[pd.Timestamp, dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        logger.warning("could not read series cache %s: %s", path, exc)
        return {}
    if loaded.empty or "requested_date" not in loaded.columns:
        return {}
    if _CACHE_SIGNATURE not in loaded.columns:
        logger.warning("series cache %s has no signature; recomputing", path)
        return {}
    table = _finalize_table(
        loaded.to_dict(orient="records"),
        ("requested_date", "error", _CACHE_SIGNATURE),
    )
    mapping: dict[pd.Timestamp, dict[str, object]] = {}
    mismatched = False
    saw_signature = False
    for record in table.to_dict(orient="records"):
        stamp = record.get("requested_date")
        if stamp is None or pd.isna(stamp):
            continue
        if not _signature_matches(record.get(_CACHE_SIGNATURE), signature):
            mismatched = True
            continue
        saw_signature = True
        record.pop(_CACHE_SIGNATURE, None)
        if not _blank_error(record.get("error")):
            continue
        mapping[_as_timestamp(stamp)] = record
    if mismatched and not saw_signature:
        logger.warning("series cache %s signature does not match; recomputing", path)
    return mapping


def _write_series_csv(path: Path, table: pd.DataFrame, signature: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = table if signature is None else table.copy()
    if signature is not None:
        frame[_CACHE_SIGNATURE] = signature
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _as_timestamp(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp) or getattr(value, "tzinfo", None) is not None:
        stamp = pd.Timestamp(value)
    else:
        parsed = parse_dates(pd.Series([value])).iloc[0]
        if pd.isna(parsed):
            raise ValueError(f"could not parse date {value!r}")
        stamp = pd.Timestamp(parsed)
    if pd.isna(stamp):
        raise ValueError(f"could not parse date {value!r}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _naive_datetimes(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values, errors="coerce")
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        return stamps.dt.tz_localize(None)
    if stamps.dtype == object:
        return stamps.map(_naive_or_nat)
    return stamps


def _naive_or_nat(value) -> pd.Timestamp:
    try:
        if value is None or pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass
    try:
        return _as_timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT


def _blank_error(value) -> bool:
    try:
        if value is None or pd.isna(value):
            return True
    except (TypeError, ValueError):
        if value is None:
            return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _signature_matches(value, signature: str) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except (TypeError, ValueError):
        if value is None:
            return False
    return str(value) == signature


def _series_signature(
    ds: Dataset,
    *,
    K: float,
    gamma: int,
    quote_set,
    days: int | None,
    target_days: int | None,
    window: tuple[int, int] | None,
    am_settlement: int | None,
    snap: str | None,
    frontier_kwargs: dict,
) -> str:
    payload = {
        "K": float(K),
        "gamma": int(gamma),
        "quote_set": _stable_token(quote_set),
        "days": None if days is None else int(days),
        "target_days": None if target_days is None else int(target_days),
        "window": None if window is None else [int(window[0]), int(window[1])],
        "am_settlement": None if am_settlement is None else int(am_settlement),
        "snap": None if snap is None else str(snap),
        "frontier_kwargs": _stable_token(frontier_kwargs),
        "dataset": _stable_token(_dataset_fingerprint(ds)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dataset_fingerprint(ds: Dataset) -> object:
    try:
        payload = ds.spec()
    except (AttributeError, TypeError, ValueError):
        return _memory_fingerprint(ds)
    if not isinstance(payload, dict):
        return _memory_fingerprint(ds)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"cache_dir", "memory_limit"}
    }


def _memory_fingerprint(ds: Dataset) -> dict[str, object]:
    source = getattr(ds, "_source", None)
    frame = getattr(source, "_frame", None)
    payload: dict[str, object] = {"kind": "memory"}
    if isinstance(frame, pd.DataFrame):
        payload["frame"] = hashlib.sha256(
            frame.to_csv(index=False).encode("utf-8")
        ).hexdigest()
    else:
        payload["dates"] = [
            _as_timestamp(item).strftime("%Y-%m-%d") for item in ds.available_dates()
        ]
    filt = getattr(ds, "_quote_filter", None)
    fields = getattr(filt, "__dataclass_fields__", None)
    if fields is not None:
        payload["quote_filter"] = {
            name: getattr(filt, name) for name in fields
        }
    return payload


def _stable_token(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number != number:
            return "nan"
        if number == float("inf"):
            return "inf"
        if number == float("-inf"):
            return "-inf"
        return number
    if isinstance(value, pd.Timestamp):
        return _as_timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return hashlib.sha256(value.to_csv(index=False).encode("utf-8")).hexdigest()
    if isinstance(value, pd.Series):
        return hashlib.sha256(value.to_csv(index=False).encode("utf-8")).hexdigest()
    if isinstance(value, np.ndarray):
        return [_stable_token(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_stable_token(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_token(value[key]) for key in sorted(value, key=str)}
    return repr(value)
