from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoxNormalizer"]


@dataclass(frozen=True)
class BoxNormalizer:
    """Affine map from a coordinate box onto the unit square.

    ``to_unit(x) = (x - origin) / scale`` and ``to_raw(u) = origin + scale * u``.
    Support values transform as ``h_unit(u) = h_raw(u / scale) - (u / scale) · origin``.
    """

    origin: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(2))
        object.__setattr__(self, "scale", np.asarray(self.scale, dtype=float).reshape(2))

    def to_unit(self, points: np.ndarray) -> np.ndarray:
        """Map raw ``(prob, loss)`` coordinates into the unit box."""
        values = np.asarray(points, dtype=float)
        return (values - self.origin) / self.scale

    def to_raw(self, points: np.ndarray) -> np.ndarray:
        """Map unit-box coordinates back to raw ``(prob, loss)``."""
        values = np.asarray(points, dtype=float)
        return self.origin + self.scale * values

    def raw_direction(self, unit_direction: np.ndarray) -> np.ndarray:
        """Pull back a unit-box direction: ``a = u / scale``."""
        return np.asarray(unit_direction, dtype=float) / self.scale

    def unit_support(self, raw_support: float, unit_direction: np.ndarray) -> float:
        """Push a raw support value forward: ``h_bar(u) = h_raw(a) - a · origin``."""
        direction = self.raw_direction(unit_direction)
        return float(raw_support) - float(direction @ self.origin)

    def raw_halfplane(
        self, unit_direction: np.ndarray, unit_support: float
    ) -> tuple[np.ndarray, float]:
        """Inverse of ``raw_direction`` / ``unit_support``: ``(a, h_raw)``."""
        direction = self.raw_direction(unit_direction)
        offset = float(unit_support) + float(direction @ self.origin)
        return direction, offset
