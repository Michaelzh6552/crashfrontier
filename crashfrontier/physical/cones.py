from __future__ import annotations

import cvxpy as cp

__all__ = ["moment_cone_order2", "moment_cone_order3", "psd2"]


def psd2(u, v, w) -> list[cp.Constraint]:
    """Second-order representation of ``[[u, v], [v, w]] ≽ 0``."""
    return [
        u >= 0.0,
        w >= 0.0,
        cp.SOC(u + w, cp.hstack([2.0 * v, u - w])),
    ]


def moment_cone_order3(y0, y1, y2, y3, a, b) -> list[cp.Constraint]:
    """Local third-order moment cone on ``[a, b]`` or ``[a, ∞)`` if ``b`` is None."""
    shifted_left = psd2(y1 - a * y0, y2 - a * y1, y3 - a * y2)
    if b is None:
        return psd2(y0, y1, y2) + shifted_left
    shifted_right = psd2(b * y0 - y1, b * y1 - y2, b * y2 - y3)
    return shifted_left + shifted_right


def moment_cone_order2(y0, y1, y2, a, b) -> list[cp.Constraint]:
    """Local second-order moment cone on ``[a, b]`` or ``[a, ∞)`` if ``b`` is None."""
    constraints = psd2(y0, y1, y2) + [y1 >= a * y0]
    if b is not None:
        constraints.extend(
            [
                y1 <= b * y0,
                y2 <= (a + b) * y1 - a * b * y0,
            ]
        )
    return constraints
