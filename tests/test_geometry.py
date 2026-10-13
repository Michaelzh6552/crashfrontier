from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from crashfrontier.geometry.depth import (
    area_ratio,
    depth_boundary,
    marginal_wedge,
    transformed_area,
)
from crashfrontier.geometry.normalize import BoxNormalizer
from crashfrontier.geometry.polygon import (
    affine_dimension,
    clip_polygon,
    convex_hull,
    halfplane_intersection,
    point_in_convex_polygon,
    polygon_area,
    segment_endpoints,
    unique_points,
)

UNIT_SQUARE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)


def _sorted_vertices(points: np.ndarray, decimals: int = 8) -> np.ndarray:
    return np.asarray(sorted(map(tuple, np.round(points, decimals))))


def _vertical_width(polygon: np.ndarray, p: float) -> float:
    losses: list[float] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        p_left = float(previous[0])
        p_right = float(current[0])
        if abs(p_right - p_left) <= 1e-15:
            continue
        lo, hi = sorted((p_left, p_right))
        if p < lo - 1e-12 or p > hi + 1e-12:
            continue
        weight = (p - p_left) / (p_right - p_left)
        losses.append(float(previous[1] + weight * (current[1] - previous[1])))
    if len(losses) < 2:
        return 0.0
    return max(losses) - min(losses)


def test_unit_square_area_is_one() -> None:
    assert polygon_area(UNIT_SQUARE) == pytest.approx(1.0)
    assert polygon_area([[0.0, 0.0], [1.0, 0.0]]) == 0.0


def test_convex_hull_is_counterclockwise_and_drops_interior() -> None:
    cloud = np.vstack(
        [UNIT_SQUARE, [[0.25, 0.25], [0.5, 0.5], [0.8, 0.2]]]
    )
    hull = convex_hull(cloud)
    assert len(hull) == 4
    signed = 0.5 * float(
        hull[:, 0] @ np.roll(hull[:, 1], -1) - hull[:, 1] @ np.roll(hull[:, 0], -1)
    )
    assert signed > 0.0
    assert np.allclose(_sorted_vertices(hull), _sorted_vertices(UNIT_SQUARE))


def test_convex_hull_returns_uniqued_degenerate_points() -> None:
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=float)
    hull = convex_hull(line)
    assert len(hull) == 3
    assert affine_dimension(hull, 1e-12) == 1
    assert len(convex_hull([[0.0, 0.0], [0.0, 0.0]])) == 1


def test_clip_polygon_halfplane_cuts_unit_square() -> None:
    clipped = clip_polygon(UNIT_SQUARE, np.array([1.0, 0.0]), 0.5)
    assert polygon_area(clipped) == pytest.approx(0.5)
    assert np.all(clipped[:, 0] <= 0.5 + 1e-10)
    empty = clip_polygon(UNIT_SQUARE, np.array([1.0, 0.0]), -1.0)
    assert len(empty) == 0


def test_clip_polygon_does_not_extrapolate_edges() -> None:
    clipped = clip_polygon(
        UNIT_SQUARE,
        np.array([1.0, 0.0]),
        upper=1.0 - 2e-12,
        tolerance=1e-12,
    )
    assert len(clipped) == 4
    assert np.all(clipped >= -1e-15)
    assert np.all(clipped <= 1.0 + 1e-15)


def test_halfplane_intersection_recovers_square() -> None:
    normals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    offsets = np.array([1.0, 1.0, 0.0, 0.0])
    recovered = halfplane_intersection(
        normals, offsets, interior_points=UNIT_SQUARE
    )
    assert np.allclose(_sorted_vertices(recovered), _sorted_vertices(UNIT_SQUARE))
    assert polygon_area(recovered) == pytest.approx(1.0, abs=1e-8)
    for corner in UNIT_SQUARE:
        assert point_in_convex_polygon(corner, recovered, tolerance=1e-8)


def test_halfplane_intersection_recovers_triangle() -> None:
    root_two = math.sqrt(2.0)
    normals = np.array(
        [[-1.0, 0.0], [0.0, -1.0], [1.0 / root_two, 1.0 / root_two]]
    )
    offsets = np.array([0.0, 0.0, 1.0 / root_two])
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    recovered = halfplane_intersection(
        normals, offsets, interior_points=triangle
    )
    assert np.allclose(_sorted_vertices(recovered), _sorted_vertices(triangle))


def test_halfplane_intersection_unbounded_raises() -> None:
    normals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    offsets = np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="unbounded"):
        halfplane_intersection(normals, offsets, interior_points=np.array([[0.5, 0.5]]))


def test_point_in_convex_polygon_interior_and_exterior() -> None:
    assert point_in_convex_polygon(np.array([0.5, 0.5]), UNIT_SQUARE)
    assert point_in_convex_polygon(np.array([0.0, 0.0]), UNIT_SQUARE)
    assert not point_in_convex_polygon(np.array([1.2, 0.5]), UNIT_SQUARE)
    assert point_in_convex_polygon(
        np.array([0.4, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    assert not point_in_convex_polygon(
        np.array([0.4, 0.2]), np.array([[0.0, 0.0], [1.0, 0.0]])
    )


def test_short_segment_contains_uses_euclidean_tolerance() -> None:
    segment = np.array([[0.0, 0.0], [1e-8, 0.0]], dtype=float)
    tolerance = 1e-9
    overhang = np.array([1e-8 + 5e-10, 0.0], dtype=float)
    assert point_in_convex_polygon(overhang, segment, tolerance=tolerance)
    assert not point_in_convex_polygon(
        np.array([1e-8 + 5e-9, 0.0], dtype=float),
        segment,
        tolerance=tolerance,
    )


def test_unique_points_affine_dimension_and_segment_endpoints() -> None:
    cloud = np.array(
        [[0.0, 0.0], [1e-12, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]],
        dtype=float,
    )
    uniqued = unique_points(cloud, 1e-9)
    assert len(uniqued) == 4
    assert affine_dimension([[0.0, 0.0]], 1e-12) == 0
    assert affine_dimension(cloud, 1e-9) == 1
    assert affine_dimension(UNIT_SQUARE, 1e-12) == 2
    ends = segment_endpoints(cloud)
    assert {tuple(row) for row in np.round(ends, 12)} == {(0.0, 0.0), (2.0, 0.0)}


def test_transformed_area_matches_rectangle_closed_form() -> None:
    rectangle = np.array(
        [[0.1, 0.01], [0.2, 0.01], [0.2, 0.02], [0.1, 0.02]],
        dtype=float,
    )
    area, low, high = transformed_area(rectangle, 1e-4)
    assert area == pytest.approx(0.01 * math.log(2.0), rel=0.0, abs=1e-12)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.2)


def test_transformed_area_matches_quad_on_random_triangles() -> None:
    rng = np.random.default_rng(0)
    p0 = 1e-4
    compared = 0
    for _ in range(20):
        triangle = np.column_stack(
            [
                rng.uniform(0.05, 0.8, size=3),
                rng.uniform(0.001, 0.2, size=3),
            ]
        )
        if polygon_area(triangle) < 1e-8:
            continue
        area, _, _ = transformed_area(triangle, p0)
        p_lo = max(p0, float(np.min(triangle[:, 0])))
        p_hi = float(np.max(triangle[:, 0]))

        def integrand(p: float, vertices: np.ndarray = triangle) -> float:
            width = _vertical_width(vertices, p)
            return 0.0 if p <= 0.0 else width / p

        numeric, _ = quad(integrand, p_lo, p_hi, epsabs=1e-12)
        assert area == pytest.approx(numeric, rel=0.0, abs=1e-8)
        compared += 1
    assert compared >= 10


def test_transformed_area_across_zero_is_finite_or_infinite() -> None:
    through_origin = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    area, low, high = transformed_area(through_origin, 0.0)
    assert area == pytest.approx(1.0, abs=1e-12)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)

    touching = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    area, _, _ = transformed_area(touching, 0.0)
    assert math.isinf(area)

    crossing = np.array(
        [[-0.2, 0.0], [0.3, 0.0], [0.3, 0.1], [-0.2, 0.1]],
        dtype=float,
    )
    area, _, _ = transformed_area(crossing, -1.0)
    assert math.isinf(area)


def test_transformed_area_keeps_tiny_nonzero_singularity() -> None:
    height = 1e-13
    crossing = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [1.0, height], [-1.0, height]],
        dtype=float,
    )
    area, _, _ = transformed_area(crossing, 0.0)
    assert math.isinf(area)


def test_transformed_area_degenerate_is_nan() -> None:
    area, low, high = transformed_area([[0.2, 0.01], [0.3, 0.02]], 1e-4)
    assert math.isnan(area) and math.isnan(low) and math.isnan(high)


def test_depth_boundary_maps_and_closes() -> None:
    rectangle = np.array(
        [[0.1, 0.01], [0.2, 0.01], [0.2, 0.02], [0.1, 0.02]],
        dtype=float,
    )
    boundary = depth_boundary(rectangle, samples_per_edge=8)
    assert len(boundary) == 8 * 4 + 1
    assert np.allclose(boundary[0], boundary[-1])
    assert np.all(boundary[:, 0] > 0.0)
    interior = boundary[:-1]
    reconstructed = interior[:, 0] * interior[:, 1]
    assert reconstructed.min() >= 0.01 - 1e-12
    assert reconstructed.max() <= 0.02 + 1e-12


def test_depth_boundary_clips_before_dividing_by_small_probability() -> None:
    polygon = np.array(
        [[1e-9, -2e-9], [0.2, 0.01], [0.2, 0.02], [1e-9, 0.0]],
        dtype=float,
    )
    boundary = depth_boundary(polygon, samples_per_edge=16, p0=1e-4)
    assert len(boundary) > 0
    assert np.min(boundary[:, 0]) >= 1e-4 - 2e-10
    assert np.min(boundary[:, 1]) > -3e-5


def test_marginal_wedge_and_area_ratio() -> None:
    lo = np.array([0.2, 0.01])
    hi = np.array([0.4, 0.05])
    polygon, area = marginal_wedge(lo, hi, K=0.9, p0=1e-4)
    assert polygon_area(polygon) > 0.0
    assert area == pytest.approx(0.04 * math.log(2.0), rel=0.0, abs=1e-10)
    assert area_ratio(area, area) == pytest.approx(1.0)
    assert math.isnan(area_ratio(1.0, 0.0))
    assert math.isnan(area_ratio(1.0, -1.0))

    cut_lo = np.array([0.05, 0.0])
    cut_hi = np.array([0.4, 0.2])
    wedge, wedge_area = marginal_wedge(cut_lo, cut_hi, K=0.9, p0=1e-4)
    assert np.all(wedge[:, 1] >= -1e-12)
    assert np.all(wedge[:, 1] <= 0.9 * wedge[:, 0] + 1e-9)
    assert wedge_area > 0.0
    assert np.isfinite(wedge_area)


def test_box_normalizer_roundtrip_direction_and_halfplane() -> None:
    normalizer = BoxNormalizer(origin=np.array([-1.0, 2.0]), scale=np.array([4.0, 5.0]))
    raw = np.array([[1.0, 7.0], [3.0, 2.0]])
    unit = normalizer.to_unit(raw)
    assert np.allclose(unit, [[0.5, 1.0], [1.0, 0.0]])
    assert np.allclose(normalizer.to_raw(unit), raw)

    direction = np.array([0.6, 0.8])
    raw_direction = normalizer.raw_direction(direction)
    assert np.allclose(raw_direction, direction / np.array([4.0, 5.0]))
    raw_support = 3.0
    unit_support = normalizer.unit_support(raw_support, direction)
    assert unit_support == pytest.approx(raw_support - float(raw_direction @ normalizer.origin))
    recovered_direction, recovered_support = normalizer.raw_halfplane(
        direction, unit_support
    )
    assert np.allclose(recovered_direction, raw_direction)
    assert recovered_support == pytest.approx(raw_support)

    single = np.array([1.0, 7.0])
    assert normalizer.to_unit(single).shape == (2,)
    assert np.allclose(normalizer.to_raw(normalizer.to_unit(single)), single)
