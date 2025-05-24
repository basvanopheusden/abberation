"""Helpers for analysing ray focusing."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from . import optics, params

FloatArray = np.ndarray


def distance_to_focus(
    slope: FloatArray,
    intercept: FloatArray,
    focal_point: Tuple[float, float] = (1.2, 0.0),
) -> FloatArray:
    """Return the perpendicular distance from rays to ``focal_point``."""
    x0, y0 = focal_point
    slope = np.asarray(slope)
    intercept = np.asarray(intercept)
    return np.abs(slope * x0 + intercept - y0) / np.sqrt(slope**2 + 1)


def total_distance_to_focus(
    t: float,
    max_in_angle: float,
    focal_point: Tuple[float, float] = (1.2, 0.0),
) -> float:
    """Return the summed distance of all rays to ``focal_point``."""
    in_angles = np.linspace(max_in_angle, -max_in_angle, params.n_rays)
    _, slopes, intercepts, _ = optics.compute_frame(t, incoming_angles=in_angles)
    return float(np.sum(distance_to_focus(slopes, intercepts, focal_point)))


def find_optimal_max_in_angle(
    t: float,
    focal_point: Tuple[float, float] = (1.2, 0.0),
    search_angles: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return ``max_in_angle`` that minimizes :func:`total_distance_to_focus`."""
    if search_angles is None:
        search_angles = np.linspace(0.0, 0.6, 50)

    best_angle = float(search_angles[0])
    min_dist = float('inf')
    for angle in search_angles:
        dist = total_distance_to_focus(t, angle, focal_point)
        if dist < min_dist:
            min_dist = dist
            best_angle = float(angle)
    return best_angle, min_dist
