from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from crashfrontier.geometry.polygon import clip_polygon

__all__ = [
    "area_ratio",
    "depth_boundary",
    "marginal_wedge",
    "transformed_area",
]

_BREAKPOINT_GAP = 1e-14
_SLICE_COVERAGE = 1e-12
_VERTICAL_EDGE = 1e-14
_SINGULAR_INTERCEPT_FACTOR = 64.0


def _line_coefficients(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    slope = (right[1] - left[1]) / (right[0] - left[0])
    intercept = left[1] - slope * left[0]
    return float(slope), float(intercept)


def transformed_area(
    polygon: Sequence[np.ndarray] | np.ndarray, p0: float
) -> tuple[float, float, float]:
    """Closed-form area of ``(p, L/p)`` over the polygon clipped to ``p >= p0``.

    Returns
    -------
    area, depth_lo, depth_hi
        Transformed area and the min/max of ``L/p`` at vertices with ``p > 0``.
        Degenerate clips return ``(nan, nan, nan)``. An integral that crosses
        ``p = 0`` with a nonzero vertical width returns ``inf``.
    """
    clipped = clip_polygon(
        polygon,
        np.array([-1.0, 0.0]),
        -float(p0),
        tolerance=0.0,
    )
    if len(clipped) < 3:
        return float("nan"), float("nan"), float("nan")
    points = np.asarray(clipped, dtype=float)
    breakpoints = sorted(set(float(value) for value in points[:, 0]))
    area = 0.0
    # closed form of the transformed area, one term per breakpoint interval
    for p_left, p_right in zip(breakpoints[:-1], breakpoints[1:]):
        if p_right - p_left <= _BREAKPOINT_GAP:
            continue
        midpoint = 0.5 * (p_left + p_right)
        segments: list[tuple[float, float, float]] = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            lo, hi = sorted((float(previous[0]), float(current[0])))
            if (
                abs(current[0] - previous[0]) <= _VERTICAL_EDGE
                or midpoint < lo - _SLICE_COVERAGE
                or midpoint > hi + _SLICE_COVERAGE
            ):
                continue
            slope, intercept = _line_coefficients(previous, current)
            segments.append((slope * midpoint + intercept, slope, intercept))
        if len(segments) < 2:
            continue
        lower, upper = min(segments), max(segments)
        contribution = _transformed_interval(
            p_left,
            p_right,
            upper[1] - lower[1],
            upper[2] - lower[2],
        )
        if math.isinf(contribution):
            area = float("inf")
            break
        area += contribution
    positive = points[:, 0] > 0.0
    if not np.any(positive):
        return float(area), float("nan"), float("nan")
    ratios = points[positive, 1] / points[positive, 0]
    return float(area), float(np.min(ratios)), float(np.max(ratios))


def _transformed_interval(
    p_left: float, p_right: float, delta_slope: float, delta_intercept: float
) -> float:
    """Integral of ``(Δβ + Δα / p)`` on one breakpoint interval.

    The integrand is the vertical width divided by ``p``. It is finite across
    ``p = 0`` only when the intercept gap ``Δα`` vanishes; otherwise the
    logarithmic singularity is reported as ``inf``.
    """
    width = p_right - p_left
    if p_left <= 0.0 <= p_right:
        scale = max(
            1.0,
            abs(float(delta_slope) * float(p_left)),
            abs(float(delta_slope) * float(p_right)),
        )
        singular_tolerance = (
            _SINGULAR_INTERCEPT_FACTOR * np.finfo(float).eps * scale
        )
        if abs(delta_intercept) > singular_tolerance:
            return float("inf")
        return float(delta_slope) * width
    return float(delta_slope) * width + float(delta_intercept) * math.log(p_right / p_left)


def depth_boundary(
    polygon: Sequence[np.ndarray] | np.ndarray,
    samples_per_edge: int = 80,
    p0: float = 0.0,
) -> np.ndarray:
    """Sampled closed image under ``(p, L) -> (p, L/p)`` for ``p >= p0``."""
    floor = max(float(p0), 0.0)
    vertices = np.asarray(
        clip_polygon(
            polygon,
            np.array([-1.0, 0.0]),
            -floor,
            tolerance=0.0,
        ),
        dtype=float,
    )
    if vertices.size and floor > 0.0:
        vertices[:, 0] = np.maximum(vertices[:, 0], floor)
    if vertices.size == 0 or samples_per_edge < 1:
        return np.empty((0, 2), dtype=float)
    if vertices.ndim == 1:
        vertices = vertices.reshape(1, 2)
    if len(vertices) >= 2 and np.linalg.norm(vertices[0] - vertices[-1]) == 0.0:
        vertices = vertices[:-1]
    mapped: list[np.ndarray] = []
    for index, left in enumerate(vertices):
        right = vertices[(index + 1) % len(vertices)]
        weights = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        edge = (1.0 - weights[:, None]) * left + weights[:, None] * right
        edge = edge[edge[:, 0] > 0.0]
        if len(edge):
            mapped.append(np.column_stack([edge[:, 0], edge[:, 1] / edge[:, 0]]))
    if not mapped:
        return np.empty((0, 2), dtype=float)
    boundary = np.vstack(mapped)
    return np.vstack([boundary, boundary[0]])


def marginal_wedge(
    lo: np.ndarray, hi: np.ndarray, K: float, p0: float
) -> tuple[np.ndarray, float]:
    """Rectangle ``[max(lo_p, p0), hi_p] x [lo_L, hi_L]`` cut by ``0 <= L <= K p``."""
    lower = np.asarray(lo, dtype=float).reshape(2)
    upper = np.asarray(hi, dtype=float).reshape(2)
    p_lo = max(float(lower[0]), float(p0))
    polygon = np.array(
        [
            [p_lo, lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [p_lo, upper[1]],
        ],
        dtype=float,
    )
    polygon = clip_polygon(polygon, np.array([-float(K), 1.0]), 0.0)
    polygon = clip_polygon(polygon, np.array([0.0, -1.0]), 0.0)
    area, _, _ = transformed_area(polygon, p0)
    return polygon, float(area)


def area_ratio(area: float, baseline_area: float) -> float:
    """``area / baseline_area``, or ``nan`` when the baseline is not positive."""
    baseline = float(baseline_area)
    if baseline <= 0.0:
        return float("nan")
    return float(area) / baseline
