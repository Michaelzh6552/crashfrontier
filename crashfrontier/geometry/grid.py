from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crashfrontier.geometry.polygon import convex_hull, halfplane_intersection

__all__ = ["GridResult", "enumerate_by_directions"]

_DIRECTION_KEY_DECIMALS = 10

Oracle = Callable[[np.ndarray], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class GridResult:
    """Direction-grid inner/outer reconstruction in unit-box coordinates."""

    inner: np.ndarray
    outer: np.ndarray
    level: int
    max_gap: float
    supports: pd.DataFrame
    status: str


def _direction_key(direction: np.ndarray) -> tuple[float, float]:
    return (
        round(float(direction[0]), _DIRECTION_KEY_DECIMALS),
        round(float(direction[1]), _DIRECTION_KEY_DECIMALS),
    )


def _equispaced_directions(count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, int(count), endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _max_dot(directions: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(vertices, dtype=float))
    return np.max(
        directions[:, 0, None] * points[None, :, 0]
        + directions[:, 1, None] * points[None, :, 1],
        axis=1,
    )


def _edge_normals(inner: np.ndarray) -> np.ndarray:
    """Outward unit normals of a convex polygon, one per nonzero edge."""
    points = np.atleast_2d(np.asarray(inner, dtype=float))
    if len(points) <= 1:
        return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float)
    normals: list[list[float]] = []
    for index, left in enumerate(points):
        right = points[(index + 1) % len(points)]
        edge = right - left
        length = float(np.linalg.norm(edge))
        if length <= 1e-15:
            continue
        normals.append([float(edge[1] / length), float(-edge[0] / length)])
    if not normals:
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    return np.asarray(normals, dtype=float)


def _max_support_gap(inner: np.ndarray, outer: np.ndarray) -> float:
    directions = _edge_normals(inner)
    if len(inner) == 0 or len(np.atleast_2d(outer)) == 0:
        return 0.0
    return float(
        np.max(np.maximum(0.0, _max_dot(directions, outer) - _max_dot(directions, inner)))
    )


def enumerate_by_directions(
    oracle: Oracle,
    *,
    levels: Sequence[int] = (32, 64, 128),
    max_gap: float = 0.01,
) -> GridResult:
    """Rebuild a convex set by doubling an equispaced direction grid."""
    cache: dict[tuple[float, float], tuple[np.ndarray, float]] = {}
    result: GridResult | None = None
    for level in levels:
        directions = _equispaced_directions(level)
        points = np.empty((level, 2), dtype=float)
        supports = np.empty(level, dtype=float)
        for index, direction in enumerate(directions):
            key = _direction_key(direction)
            if key not in cache:
                point, bound = oracle(direction)
                cache[key] = (
                    np.asarray(point, dtype=float).reshape(2).copy(),
                    float(bound),
                )
            point, bound = cache[key]
            points[index] = point
            supports[index] = bound
        inner = convex_hull(points)
        # enlarge, never shrink, so the outer set stays valid
        offsets = np.maximum(supports, _max_dot(directions, points))
        outer = halfplane_intersection(
            directions, offsets, interior_points=inner
        )
        gap = _max_support_gap(inner, outer)
        frame = pd.DataFrame(
            {
                "direction_1": directions[:, 0],
                "direction_2": directions[:, 1],
                "point_1": points[:, 0],
                "point_2": points[:, 1],
                "support": supports,
            }
        )
        status = "converged" if gap <= max_gap else "max_level"
        result = GridResult(
            inner=inner,
            outer=outer,
            level=int(level),
            max_gap=gap,
            supports=frame,
            status=status,
        )
        if gap <= max_gap:
            return result
    if result is None:
        raise ValueError("levels must contain at least one direction count")
    return result
