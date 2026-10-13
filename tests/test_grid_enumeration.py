from __future__ import annotations

import numpy as np
import pytest

from crashfrontier.geometry.grid import GridResult, enumerate_by_directions
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.geometry.polygon import point_in_convex_polygon, polygon_area

TRIANGLE = np.array(
    [
        [0.10, 0.12],
        [0.88, 0.18],
        [0.42, 0.86],
    ],
    dtype=float,
)
LEVELS = (32, 64, 128)


def _sorted_vertices(points: np.ndarray, decimals: int = 8) -> np.ndarray:
    return np.asarray(sorted(map(tuple, np.round(points, decimals))))


def _vertex_oracle(vertices: np.ndarray):
    cloud = np.asarray(vertices, dtype=float)

    def oracle(direction: np.ndarray) -> tuple[np.ndarray, float]:
        vector = np.asarray(direction, dtype=float).reshape(2)
        values = cloud @ vector
        index = int(np.argmax(values))
        return cloud[index].copy(), float(values[index])

    return oracle


def _edge_normal_gap(inner: np.ndarray, outer: np.ndarray) -> float:
    points = np.atleast_2d(np.asarray(inner, dtype=float))
    normals: list[list[float]] = []
    for index, left in enumerate(points):
        right = points[(index + 1) % len(points)]
        edge = right - left
        length = float(np.linalg.norm(edge))
        if length <= 1e-15:
            continue
        normals.append([float(edge[1] / length), float(-edge[0] / length)])
    directions = np.asarray(normals, dtype=float)
    inner_points = points
    outer_points = np.atleast_2d(np.asarray(outer, dtype=float))
    inner_support = np.max(inner_points @ directions.T, axis=0)
    outer_support = np.max(outer_points @ directions.T, axis=0)
    return float(np.max(np.maximum(0.0, outer_support - inner_support)))


def test_enumerate_by_directions_recovers_known_triangle() -> None:
    result = enumerate_by_directions(_vertex_oracle(TRIANGLE), levels=LEVELS)
    assert isinstance(result, GridResult)
    assert result.status in {"converged", "max_level"}
    assert result.level in LEVELS
    assert len(result.inner) == 3
    assert np.allclose(_sorted_vertices(result.inner), _sorted_vertices(TRIANGLE))
    for vertex in TRIANGLE:
        assert point_in_convex_polygon(vertex, result.outer, tolerance=1e-8)
    assert polygon_area(result.outer) + 1e-12 >= polygon_area(result.inner)
    assert list(result.supports.columns) == [
        "direction_1",
        "direction_2",
        "point_1",
        "point_2",
        "support",
    ]
    assert len(result.supports) == result.level


def test_max_gap_is_nonincreasing_across_levels() -> None:
    oracle = _vertex_oracle(TRIANGLE)
    gaps: list[float] = []
    for index in range(len(LEVELS)):
        prefix = LEVELS[: index + 1]
        result = enumerate_by_directions(oracle, levels=prefix, max_gap=0.0)
        assert result.status == "max_level"
        assert result.level == prefix[-1]
        gaps.append(result.max_gap)
    assert gaps[0] >= gaps[1] >= gaps[2]
    assert gaps[0] > 0.0


def test_direction_reuse_on_doubling_levels() -> None:
    inner_oracle = _vertex_oracle(TRIANGLE)
    calls: list[np.ndarray] = []

    def oracle(direction: np.ndarray) -> tuple[np.ndarray, float]:
        vector = np.asarray(direction, dtype=float).reshape(2)
        calls.append(vector.copy())
        return inner_oracle(vector)

    result = enumerate_by_directions(oracle, levels=LEVELS, max_gap=0.0)
    assert result.level == 128
    assert len(calls) == 128
    keys = [
        (round(float(direction[0]), 10), round(float(direction[1]), 10))
        for direction in calls
    ]
    assert len(set(keys)) == 128


def test_max_gap_matches_edge_normals() -> None:
    result = enumerate_by_directions(
        _vertex_oracle(TRIANGLE), levels=(32,), max_gap=0.0
    )
    assert result.level == 32
    expected = _edge_normal_gap(result.inner, result.outer)
    assert result.max_gap == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert result.max_gap >= 0.0


def test_enumerate_by_directions_with_box_normalizer() -> None:
    raw = np.array([[0.20, 0.02], [0.55, 0.03], [0.35, 0.12]], dtype=float)
    origin = raw.min(axis=0)
    scale = raw.max(axis=0) - origin
    normalizer = BoxNormalizer(origin, scale)

    def oracle(unit_direction: np.ndarray) -> tuple[np.ndarray, float]:
        raw_direction = normalizer.raw_direction(unit_direction)
        values = raw @ raw_direction
        point = raw[int(np.argmax(values))]
        return (
            normalizer.to_unit(point),
            normalizer.unit_support(float(np.max(values)), unit_direction),
        )

    result = enumerate_by_directions(oracle, levels=LEVELS)
    recovered = normalizer.to_raw(result.inner)
    assert np.allclose(_sorted_vertices(recovered), _sorted_vertices(raw))
    assert result.max_gap >= 0.0
