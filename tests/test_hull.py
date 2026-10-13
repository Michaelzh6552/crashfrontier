from __future__ import annotations

import math

import numpy as np

from crashfrontier.geometry.hull import HullResult, enumerate_hull
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


def test_enumerate_hull_recovers_known_triangle() -> None:
    result = enumerate_hull(_vertex_oracle(TRIANGLE))
    assert isinstance(result, HullResult)
    assert result.status == "edges_certified"
    assert result.dimension == 2
    assert len(result.inner) == 3
    assert np.allclose(_sorted_vertices(result.inner), _sorted_vertices(TRIANGLE))
    signed = 0.5 * float(
        result.inner[:, 0] @ np.roll(result.inner[:, 1], -1)
        - result.inner[:, 1] @ np.roll(result.inner[:, 0], -1)
    )
    assert signed > 0.0
    assert result.max_gap <= 1e-8
    assert result.calls >= 8
    assert len(result.certificates) == result.calls
    for vertex in result.inner:
        assert point_in_convex_polygon(vertex, result.outer, tolerance=1e-8)
    assert polygon_area(result.outer) + 1e-12 >= polygon_area(result.inner)


def test_enumerate_hull_point_degeneracy() -> None:
    point = np.array([0.31, 0.44], dtype=float)
    result = enumerate_hull(_vertex_oracle(point.reshape(1, 2)))
    assert result.status == "point"
    assert result.dimension == 0
    assert result.max_gap == 0.0
    assert len(result.inner) == 1
    assert np.allclose(result.inner[0], point)
    assert np.allclose(result.outer, result.inner)
    assert result.calls == 8
    assert len(result.certificates) == 8


def test_enumerate_hull_segment_degeneracy() -> None:
    segment = np.array([[0.20, 0.30], [0.80, 0.45]], dtype=float)
    result = enumerate_hull(_vertex_oracle(segment))
    assert result.status == "segment"
    assert result.dimension == 1
    assert result.max_gap == 0.0
    assert len(result.inner) == 2
    assert np.allclose(_sorted_vertices(result.inner), _sorted_vertices(segment))
    assert np.allclose(result.outer, result.inner)
    assert result.calls == 10
    assert len(result.certificates) == 10


def test_enumerate_hull_certificates_are_valid_halfplanes() -> None:
    seen: list[np.ndarray] = []

    inner_oracle = _vertex_oracle(TRIANGLE)

    def oracle(direction: np.ndarray) -> tuple[np.ndarray, float]:
        vector = np.asarray(direction, dtype=float).reshape(2)
        assert vector.shape == (2,)
        assert abs(float(np.linalg.norm(vector)) - 1.0) <= 1e-12
        seen.append(vector.copy())
        return inner_oracle(vector)

    result = enumerate_hull(oracle)
    assert result.status == "edges_certified"
    assert len(seen) == result.calls
    for direction, bound in result.certificates:
        assert abs(float(np.linalg.norm(direction)) - 1.0) <= 1e-12
        assert float(np.max(result.inner @ direction)) <= bound + 1e-10


def test_enumerate_hull_area_certified_and_budget_reached() -> None:
    certified = enumerate_hull(
        _vertex_oracle(TRIANGLE),
        area_predicate=lambda inner, outer: True,
    )
    assert certified.status == "area_certified"
    assert certified.dimension == 2
    assert certified.calls == 8
    assert math.isnan(certified.max_gap)

    budgeted = enumerate_hull(_vertex_oracle(TRIANGLE), max_calls=8)
    assert budgeted.status == "budget_reached"
    assert budgeted.calls == 8
    assert budgeted.dimension == 2
    assert math.isfinite(budgeted.max_gap)
    assert budgeted.max_gap >= 0.0


def test_enumerate_hull_stalled_when_support_is_inconsistent() -> None:
    vertices = TRIANGLE

    def oracle(direction: np.ndarray) -> tuple[np.ndarray, float]:
        vector = np.asarray(direction, dtype=float).reshape(2)
        values = vertices @ vector
        index = int(np.argmax(values))
        return vertices[index].copy(), float(values[index]) + 1.0

    result = enumerate_hull(oracle)
    assert result.status == "stalled"
    assert result.dimension == 2
    assert result.max_gap > 1e-8


def test_enumerate_hull_with_box_normalizer() -> None:
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

    result = enumerate_hull(oracle)
    assert result.status == "edges_certified"
    recovered = normalizer.to_raw(result.inner)
    assert np.allclose(_sorted_vertices(recovered), _sorted_vertices(raw))
