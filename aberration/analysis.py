"""Helpers for analysing ray focusing."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from scipy import optimize

import numpy as np

from . import optics, params

FloatArray = np.ndarray


def distance_to_focus(
    slope: FloatArray,
    intercept: FloatArray,
    focal_point: Tuple[float, float] = params.focal_point,
) -> FloatArray:
    """Return the horizontal distance from rays to ``focal_point``.

    The distance is measured along the ``x`` axis between ``x0`` and the
    intersection of the ray with the horizontal line ``y = y0``.  Rays are
    specified by ``slope`` and ``intercept`` in the form ``y = slope * x +
    intercept``.
    """
    x0, y0 = focal_point
    slope = np.asarray(slope)
    intercept = np.asarray(intercept)

    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = np.where(slope != 0, (y0 - intercept) / slope, x0)

    return np.abs(x_cross - x0)


def total_distance_to_focus(
    t: float,
    max_in_angle: float,
    focal_point: Tuple[float, float] = params.focal_point,
) -> float:
    """Return the summed distance of all rays to ``focal_point``."""
    in_angles = np.linspace(max_in_angle, -max_in_angle, params.n_rays)
    _, slopes, intercepts, _ = optics.compute_frame(t, incoming_angles=in_angles)
    return float(np.sum(distance_to_focus(slopes, intercepts, focal_point)))


def find_optimal_max_in_angle(
    t: float,
    focal_point: Tuple[float, float] = params.focal_point,
    search_angles: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return ``max_in_angle`` that minimizes :func:`total_distance_to_focus`."""
    if search_angles is not None:
        best_angle = float(search_angles[0])
        min_dist = float("inf")
        for angle in search_angles:
            dist = total_distance_to_focus(t, angle, focal_point)
            if dist < min_dist:
                min_dist = dist
                best_angle = float(angle)
        return best_angle, min_dist

    result = optimize.minimize_scalar(
        lambda ang: total_distance_to_focus(t, float(ang), focal_point),
        bounds=(0.0, 2.0),
        method="bounded",
    )
    return float(result.x), float(result.fun)
