from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

__all__ = [
    "affine_dimension",
    "clip_polygon",
    "convex_hull",
    "halfplane_intersection",
    "point_in_convex_polygon",
    "polygon_area",
    "segment_endpoints",
    "unique_points",
]

_BOUNDING_RADIUS_PADDING = 100.0
_CENTER_PROJECTION_LIMIT = 1000
_CLIP_INTERSECTION_EPS = 1e-15
_HULL_POINT_TOLERANCE = 1e-15


def _as_points(points: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim == 1:
        if values.shape[0] != 2:
            raise ValueError("points must have shape (n, 2)")
        return values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return values


def _drop_consecutive_duplicates(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance
    values = points[keep]
    if len(values) > 1 and np.linalg.norm(values[0] - values[-1]) <= tolerance:
        values = values[:-1]
    return values


def polygon_area(points: Sequence[np.ndarray] | np.ndarray) -> float:
    """Shoelace area of a simple polygon; fewer than three vertices yield 0."""
    values = _as_points(points)
    if len(values) < 3:
        return 0.0
    return 0.5 * abs(
        float(
            values[:, 0] @ np.roll(values[:, 1], -1)
            - values[:, 1] @ np.roll(values[:, 0], -1)
        )
    )


def unique_points(
    points: Sequence[np.ndarray] | np.ndarray, tolerance: float
) -> np.ndarray:
    """Drop later points that lie within ``tolerance`` of an earlier one."""
    values = _as_points(points)
    unique: list[np.ndarray] = []
    for point in values:
        if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
            unique.append(point)
    if not unique:
        return np.empty((0, 2), dtype=float)
    return np.asarray(unique, dtype=float)


def convex_hull(points: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull vertices.

    Fewer than three points, or a Qhull degeneracy, returns the uniqued input.
    """
    values = unique_points(points, _HULL_POINT_TOLERANCE)
    if len(values) < 3:
        return values
    try:
        hull = ConvexHull(values)
    except QhullError:
        return values
    return values[hull.vertices]


def clip_polygon(
    polygon: Sequence[np.ndarray] | np.ndarray,
    normal: np.ndarray,
    upper: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Sutherland–Hodgman clip keeping ``normal·x <= upper + tolerance``."""
    vertices = _as_points(polygon)
    if len(vertices) == 0:
        return vertices
    direction = np.asarray(normal, dtype=float).reshape(2)
    clipped: list[np.ndarray] = []
    effective_upper = float(upper) + float(tolerance)
    for index, current in enumerate(vertices):
        previous = vertices[index - 1]
        current_value = float(direction @ current - effective_upper)
        previous_value = float(direction @ previous - effective_upper)
        current_inside = current_value <= 0.0
        previous_inside = previous_value <= 0.0
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > _CLIP_INTERSECTION_EPS:
                weight = float(np.clip(previous_value / denominator, 0.0, 1.0))
                clipped.append(previous + weight * (current - previous))
        if current_inside:
            clipped.append(np.asarray(current, dtype=float))
    if not clipped:
        return np.empty((0, 2), dtype=float)
    return _drop_consecutive_duplicates(np.asarray(clipped, dtype=float), tolerance)


def halfplane_intersection(
    normals: np.ndarray,
    offsets: np.ndarray,
    interior_points: np.ndarray | None = None,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Bounded intersection of half-planes ``normal_i · x <= offset_i``.

    Raises ``ValueError`` when the intersection is empty or unbounded.
    """
    directions = np.asarray(normals, dtype=float)
    bounds = np.asarray(offsets, dtype=float).reshape(-1)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("normals must be an n-by-2 array")
    if bounds.shape != (len(directions),):
        raise ValueError("offsets must match normals")
    if not np.isfinite(directions).all() or not np.isfinite(bounds).all():
        raise ValueError("half-planes must be finite")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("half-plane normals must be nonzero")
    directions = directions / lengths[:, None]
    bounds = bounds / lengths

    if interior_points is None:
        center = np.zeros(2, dtype=float)
    else:
        seeds = _as_points(interior_points)
        if len(seeds) == 0:
            raise ValueError("interior_points must be a nonempty n-by-2 array")
        center = seeds.mean(axis=0)
    for _ in range(_CENTER_PROJECTION_LIMIT):
        violations = directions @ center - bounds
        worst = int(np.argmax(violations))
        if violations[worst] <= tolerance:
            break
        center = center - (violations[worst] + 0.5 * tolerance) * directions[worst]
    else:
        raise ValueError("half-plane intersection has no feasible center")

    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)
    angles.sort()
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
    maximum_gap = float(gaps.max()) if len(gaps) else 2.0 * math.pi
    if maximum_gap >= math.pi - tolerance:
        raise ValueError("half-plane intersection is unbounded")
    slacks = bounds - directions @ center
    radius = max(tolerance, float(slacks.max()) / math.cos(0.5 * maximum_gap))
    radius *= 1.0 + _BOUNDING_RADIUS_PADDING * tolerance
    polygon = center + radius * np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    for direction, offset in zip(directions, bounds):
        polygon = clip_polygon(polygon, direction, float(offset), tolerance=tolerance)
        if len(polygon) == 0:
            raise ValueError("half-plane intersection is empty")
    return polygon


def point_in_convex_polygon(
    point: np.ndarray,
    polygon: Sequence[np.ndarray] | np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """Whether ``point`` lies in the convex hull of ``polygon``."""
    values = _as_points(polygon)
    target = np.asarray(point, dtype=float).reshape(2)
    if len(values) == 0:
        return False
    if len(values) == 1:
        return bool(np.linalg.norm(target - values[0]) <= tolerance)
    if len(values) == 2:
        edge = values[1] - values[0]
        scale = float(edge @ edge)
        if scale <= 0.0:
            return bool(np.linalg.norm(target - values[0]) <= tolerance)
        weight = float((target - values[0]) @ edge / scale)
        projection = values[0] + float(np.clip(weight, 0.0, 1.0)) * edge
        return bool(np.linalg.norm(target - projection) <= tolerance)
    try:
        hull = ConvexHull(values)
    except QhullError:
        return bool(
            np.linalg.norm(target - segment_endpoints(values)[0]) <= tolerance
            or point_in_convex_polygon(target, segment_endpoints(values), tolerance)
        )
    residuals = hull.equations[:, :-1] @ target + hull.equations[:, -1]
    return bool(np.max(residuals) <= tolerance)


def affine_dimension(
    points: Sequence[np.ndarray] | np.ndarray, tolerance: float
) -> int:
    """Affine dimension of a 2-D point set: 0, 1, or 2."""
    values = _as_points(points)
    if len(values) <= 1:
        return 0
    centered = values - values[0]
    return int(np.linalg.matrix_rank(centered, tol=tolerance))


def segment_endpoints(points: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """The pair of input points with the largest Euclidean separation."""
    values = _as_points(points)
    if len(values) <= 1:
        return values
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    left, right = np.unravel_index(np.argmax(distances), distances.shape)
    return np.asarray([values[left], values[right]], dtype=float)
