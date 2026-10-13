from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from crashfrontier.geometry.polygon import (
    affine_dimension,
    clip_polygon,
    convex_hull,
    segment_endpoints,
    unique_points,
)

__all__ = ["HullResult", "enumerate_hull"]

_INITIAL_DIRECTION_COUNT = 8
_OUTER_CLIP_TOLERANCE = 1e-12
_OUTER_POINT_TOLERANCE = 1e-15
_UNIT_SQUARE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)

Oracle = Callable[[np.ndarray], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class HullResult:
    """Adaptive support-function hull in unit-box coordinates."""

    inner: np.ndarray
    outer: np.ndarray
    dimension: int
    calls: int
    max_gap: float
    certificates: list[tuple[np.ndarray, float]]
    status: str


def _as_point(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(2).copy()


def _edge_support_gap(inner: np.ndarray, outer: np.ndarray) -> float:
    """Max outer-minus-inner support along the inner polygon's edge normals."""
    inner_points = np.atleast_2d(np.asarray(inner, dtype=float))
    outer_points = np.atleast_2d(np.asarray(outer, dtype=float))
    if len(inner_points) == 0 or len(outer_points) == 0:
        return 0.0
    if not np.isfinite(inner_points).all() or not np.isfinite(outer_points).all():
        return float("inf")
    normals: list[np.ndarray] = []
    if len(inner_points) == 1:
        return float(np.max(np.linalg.norm(outer_points - inner_points[0], axis=1)))
    for index, left in enumerate(inner_points):
        right = inner_points[(index + 1) % len(inner_points)]
        edge = right - left
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            continue
        normals.append(np.array([edge[1], -edge[0]], dtype=float) / length)
    if not normals:
        return float(np.max(np.linalg.norm(outer_points - inner_points[0], axis=1)))
    directions = np.vstack(normals)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inner_support = np.max(inner_points @ directions.T, axis=0)
        outer_support = np.max(outer_points @ directions.T, axis=0)
    if not np.isfinite(inner_support).all() or not np.isfinite(outer_support).all():
        return float("inf")
    gap = float(np.max(np.maximum(0.0, outer_support - inner_support)))
    if not math.isfinite(gap):
        return float("inf")
    return gap


def _outer_from_supports(
    certificates: list[tuple[np.ndarray, float]],
    inner: np.ndarray,
) -> np.ndarray:
    outer = _UNIT_SQUARE.copy()
    contained = np.asarray(inner, dtype=float)
    if contained.ndim == 1:
        contained = contained.reshape(1, 2)
    for normal, upper in certificates:
        direction = np.asarray(normal, dtype=float).reshape(2)
        # enlarge, never shrink, so the outer set stays valid
        bound = float(upper)
        if len(contained):
            bound = max(bound, float(np.max(contained @ direction)))
        outer = clip_polygon(
            outer, direction, bound, tolerance=_OUTER_CLIP_TOLERANCE
        )
        if len(outer) == 0:
            break
    if len(contained) == 0:
        return outer
    if len(outer) == 0:
        combined = contained
    else:
        combined = np.vstack([outer, contained])
    combined = unique_points(combined, _OUTER_POINT_TOLERANCE)
    if len(combined) >= 3:
        return convex_hull(combined)
    return combined


def enumerate_hull(
    oracle: Oracle,
    *,
    edge_tolerance: float = 1e-8,
    point_tolerance: float = 1e-9,
    max_calls: int = 500,
    area_predicate: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> HullResult:
    """Rebuild a convex set from a support oracle by certifying hull edges."""
    initial_directions = [
        np.array(
            [
                math.cos(math.pi * index / 4.0),
                math.sin(math.pi * index / 4.0),
            ],
            dtype=float,
        )
        for index in range(_INITIAL_DIRECTION_COUNT)
    ]
    points: list[np.ndarray] = []
    certificates: list[tuple[np.ndarray, float]] = []
    for direction in initial_directions:
        point, bound = oracle(direction)
        points.append(_as_point(point))
        certificates.append((direction, float(bound)))
    calls = len(initial_directions)
    unique = unique_points(points, point_tolerance)
    dimension = affine_dimension(unique, point_tolerance)
    if dimension == 0:
        vertex = unique[:1]
        return HullResult(
            inner=vertex,
            outer=vertex.copy(),
            dimension=0,
            calls=calls,
            max_gap=0.0,
            certificates=certificates,
            status="point",
        )
    if dimension == 1:
        endpoints = segment_endpoints(unique)
        segment = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(segment))
        if length > point_tolerance and calls + 2 <= max_calls:
            normal = np.array([segment[1], -segment[0]], dtype=float) / length
            for probe in (normal, -normal):
                point, bound = oracle(probe)
                points.append(_as_point(point))
                certificates.append((probe, float(bound)))
                calls += 1
            unique = unique_points(points, point_tolerance)
            dimension = affine_dimension(unique, point_tolerance)
        if dimension == 1:
            endpoints = segment_endpoints(unique)
            return HullResult(
                inner=endpoints,
                outer=endpoints.copy(),
                dimension=1,
                calls=calls,
                max_gap=0.0,
                certificates=certificates,
                status="segment",
            )

    inner = convex_hull(unique)
    outer = _outer_from_supports(certificates, inner)
    if area_predicate is not None and area_predicate(inner, outer):
        return HullResult(
            inner=inner,
            outer=outer,
            dimension=2,
            calls=calls,
            max_gap=float("nan"),
            certificates=certificates,
            status="area_certified",
        )

    max_gap = float("nan")
    while calls < max_calls:
        inner = convex_hull(unique)
        additions: list[np.ndarray] = []
        max_gap = 0.0
        all_edges_checked = True
        for index, left in enumerate(inner):
            if calls >= max_calls:
                all_edges_checked = False
                break
            right = inner[(index + 1) % len(inner)]
            edge = right - left
            length = float(np.linalg.norm(edge))
            if length <= point_tolerance:
                continue
            outward = np.array([edge[1], -edge[0]], dtype=float) / length
            point, bound = oracle(outward)
            calls += 1
            bound = max(
                float(bound),
                float(outward @ left),
                float(outward @ right),
            )
            certificates.append((outward, bound))
            gap = max(0.0, float(bound - outward @ left))
            max_gap = max(max_gap, gap)
            if gap > edge_tolerance:
                additions.append(_as_point(point))
        if not all_edges_checked:
            stacked = unique if not additions else np.vstack([unique, *additions])
            expanded = unique_points(stacked, point_tolerance)
            certified = (
                convex_hull(expanded)
                if affine_dimension(expanded, point_tolerance) >= 2
                else expanded
            )
            outer = _outer_from_supports(certificates, certified)
            max_gap = max(
                float(max_gap) if math.isfinite(max_gap) else 0.0,
                _edge_support_gap(certified, outer),
            )
            return HullResult(
                inner=certified,
                outer=outer,
                dimension=2,
                calls=calls,
                max_gap=float(max_gap),
                certificates=certificates,
                status="budget_reached",
            )
        if not additions:
            return HullResult(
                inner=inner,
                outer=_outer_from_supports(certificates, inner),
                dimension=2,
                calls=calls,
                max_gap=max_gap,
                certificates=certificates,
                status="edges_certified",
            )
        expanded = unique_points(np.vstack([unique, *additions]), point_tolerance)
        if len(expanded) == len(unique):
            return HullResult(
                inner=inner,
                outer=_outer_from_supports(certificates, inner),
                dimension=2,
                calls=calls,
                max_gap=max_gap,
                certificates=certificates,
                status="stalled",
            )
        unique = expanded
        candidate_inner = convex_hull(unique)
        candidate_outer = _outer_from_supports(certificates, candidate_inner)
        if area_predicate is not None and area_predicate(
            candidate_inner, candidate_outer
        ):
            return HullResult(
                inner=candidate_inner,
                outer=candidate_outer,
                dimension=2,
                calls=calls,
                max_gap=max_gap,
                certificates=certificates,
                status="area_certified",
            )

    certified = convex_hull(unique)
    outer = _outer_from_supports(certificates, certified)
    max_gap = max(
        float(max_gap) if math.isfinite(max_gap) else 0.0,
        _edge_support_gap(certified, outer),
    )
    return HullResult(
        inner=certified,
        outer=outer,
        dimension=2,
        calls=calls,
        max_gap=float(max_gap),
        certificates=certificates,
        status="budget_reached",
    )
