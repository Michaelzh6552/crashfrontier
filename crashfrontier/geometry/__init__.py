from __future__ import annotations

from crashfrontier.geometry.depth import (
    area_ratio,
    depth_boundary,
    marginal_wedge,
    transformed_area,
)
from crashfrontier.geometry.grid import GridResult, enumerate_by_directions
from crashfrontier.geometry.hull import HullResult, enumerate_hull
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

__all__ = [
    "BoxNormalizer",
    "GridResult",
    "HullResult",
    "affine_dimension",
    "area_ratio",
    "clip_polygon",
    "convex_hull",
    "depth_boundary",
    "enumerate_by_directions",
    "enumerate_hull",
    "halfplane_intersection",
    "marginal_wedge",
    "point_in_convex_polygon",
    "polygon_area",
    "segment_endpoints",
    "transformed_area",
    "unique_points",
]
