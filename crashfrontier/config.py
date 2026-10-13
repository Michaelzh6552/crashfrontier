from __future__ import annotations

from dataclasses import dataclass

from crashfrontier.physical.solve import ConicSettings

__all__ = [
    "COLORS",
    "DEFAULT_LM_RANGE",
    "DEFAULT_MEMORY_LIMIT",
    "DEFAULT_P0",
    "DEFAULT_SERIES_TARGET",
    "DEFAULT_SERIES_WINDOW",
    "DEFAULT_SUPPORT_CAP",
    "DEFAULT_TENORS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW_FRACTION",
    "ConicSettings",
    "GridSettings",
    "LABELS",
    "LPSettings",
    "PALETTE",
    "SERIES_CACHE_FLUSH",
    "SET_ORDER",
    "WEEKDAY_PRIORITY",
]


@dataclass(frozen=True)
class LPSettings:
    """Linear program tolerances and hull reconstruction limits."""

    primal_feasibility_tolerance: float = 1e-10
    dual_feasibility_tolerance: float = 1e-10
    time_limit_seconds: float = 5.0
    edge_tolerance: float = 1e-8
    point_tolerance: float = 1e-9
    degenerate_tolerance: float = 1e-10
    max_support_calls: int = 500
    anytime_max_support_calls: int = 160
    anytime_area_relative_tolerance: float = 0.002
    anytime_area_absolute_tolerance: float = 1e-12
    interior_cutoff: float = 1e-8
    validation_multiplier: float = 5.0


@dataclass(frozen=True)
class GridSettings:
    """Direction-grid reconstruction levels and gap tolerance."""

    levels: tuple[int, ...] = (32, 64, 128)
    max_normalized_gap: float = 0.01


DEFAULT_THRESHOLD = 0.90
DEFAULT_P0 = 1e-4
DEFAULT_SUPPORT_CAP = 5.0
DEFAULT_LM_RANGE = (-0.35, 0.05)
DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024
DEFAULT_TENORS = (7, 14, 30, 60, 90)
DEFAULT_WINDOW_FRACTION = 0.35
DEFAULT_SERIES_TARGET = 30
DEFAULT_SERIES_WINDOW = (20, 45)
SERIES_CACHE_FLUSH = 20
WEEKDAY_PRIORITY = (2, 1, 3, 0, 4)
SET_ORDER = ("put_pair", "local_eight", "put_wing", "full_otm", "full_panel")
COLORS = {
    "put_pair": "#2C4E86",
    "local_eight": "#0F7F78",
    "put_wing": "#A66B00",
    "full_otm": "#2C7A45",
    "full_panel": "#8E3D5B",
}
LABELS = {
    "put_pair": "Put pair",
    "local_eight": "Local eight",
    "put_wing": "Complete put wing",
    "full_otm": "All OTM quotes",
    "full_panel": "All quotes",
}
# Coordinated unknowns. Shifted from COLORS and from quote-role hues so an
# unnamed series is not mistaken for a named information set or a bid/ask role.
PALETTE = (
    "#355C7D",
    "#3E7C74",
    "#8C6A2F",
    "#3E6B4A",
    "#7A4E62",
    "#5C6570",
    "#6E5344",
)
