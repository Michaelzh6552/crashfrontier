from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from crashfrontier.frontier import Frontier
from crashfrontier.riskneutral.certificates import DualCertificate

__all__ = ["frontier_frames", "write_frontier", "write_json"]

_TABLE_NAMES = ("summary", "vertices", "halfplanes", "quotes")


def frontier_frames(fronts: Frontier | dict[str, Frontier]) -> dict[str, pd.DataFrame]:
    """Stack one or more frontiers into summary, vertex, half-plane, and quote tables."""
    items = _as_named(fronts)
    summaries: list[pd.DataFrame] = []
    vertices: list[pd.DataFrame] = []
    halfplanes: list[pd.DataFrame] = []
    quotes: list[pd.DataFrame] = []
    for name, frontier in items:
        summary = pd.DataFrame([frontier.summary()])
        summary.insert(0, "name", name)
        summaries.append(summary)
        vertex_table = frontier.vertices()
        vertex_table.insert(0, "name", name)
        vertices.append(vertex_table)
        plane_table = frontier.halfplane_table()
        plane_table.insert(0, "name", name)
        halfplanes.append(plane_table)
        quote_table = frontier.quotes.copy()
        quote_table.insert(0, "name", name)
        quotes.append(quote_table)
    return {
        "summary": _concat(summaries),
        "vertices": _concat(vertices),
        "halfplanes": _concat(halfplanes),
        "quotes": _concat(quotes),
    }


def write_frontier(
    fronts: Frontier | dict[str, Frontier],
    path_stem: str | Path,
    *,
    formats: tuple[str, ...] = ("csv", "json"),
    certificate: DualCertificate | None = None,
) -> list[Path]:
    """Write frontier tables as CSV and/or a JSON bundle next to ``path_stem``."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    chosen = tuple(str(item) for item in formats)
    written: list[Path] = []
    frames = frontier_frames(fronts)
    if "csv" in chosen:
        for name in _TABLE_NAMES:
            path = Path(f"{stem}_{name}.csv")
            frames[name].to_csv(path, index=False)
            written.append(path)
        if certificate is not None:
            path = Path(f"{stem}_binding.csv")
            certificate.to_frame().to_csv(path, index=False)
            written.append(path)
    if "json" in chosen:
        payload = {
            "frontiers": {name: frontier.to_dict() for name, frontier in _as_named(fronts)},
            "certificate": None
            if certificate is None
            else certificate.to_frame().to_dict(orient="records"),
        }
        path = Path(f"{stem}.json")
        write_json(payload, path)
        written.append(path)
    return written


def write_json(obj, path) -> None:
    """Atomically write ``obj`` as JSON, accepting ndarray / Timestamp / numpy scalars."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    payload = _json_safe(obj)
    temporary.write_text(
        json.dumps(payload, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _as_named(fronts: Frontier | dict[str, Frontier]) -> list[tuple[str, Frontier]]:
    if isinstance(fronts, Frontier):
        return [(str(fronts.quote_set), fronts)]
    return [(str(name), frontier) for name, frontier in fronts.items()]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value):
    """Replace non-finite floats and missing timestamps with ``None``."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
