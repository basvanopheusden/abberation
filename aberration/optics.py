"""Core optical computations."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from . import params

FloatArray = np.ndarray


def _surface_radius(t: float) -> float:
    """Return the radius of curvature for interpolation parameter ``t``."""
    start_angle = np.arcsin(np.clip(params.aperture / params.radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(params.aperture / params.far_radius)
    angle = (1 - t) * start_angle + t * end_angle
    return params.aperture / np.sin(angle)


def surface_coordinates(t: float) -> FloatArray:
    """Return x coordinates of the optical surface for parameter ``t``."""
    r = _surface_radius(t)
    return np.where(
        np.abs(params.surf_y) <= r,
        np.sqrt(r**2 - params.surf_y**2) - r + params.plane_x,
        np.nan,
    )


def ray_parameters(
    t: float,
    n_ratio: float = params.ref_index_ratio,
    incoming_angles: Optional[Sequence[float]] = None,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Return slopes, intercepts and surface intersections for rays."""
    r = _surface_radius(t)
    if incoming_angles is None:
        incoming_angles = params.incoming_final_angles
    slopes = []
    intercepts = []
    kinks = []
    for idx, y in enumerate(params.ys):
        if np.abs(y) <= r:
            x_int = np.sqrt(r**2 - y**2) - r + params.plane_x
            normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        else:
            x_int = np.nan
            normal_angle = 0.0
        inc_angle = t * incoming_angles[idx]
        phi_in = np.abs(inc_angle - normal_angle)
        phi_out = np.arcsin(
            np.clip(np.sin(phi_in) * n_ratio, -1 + 1e-9, 1 - 1e-9)
        )
        orient = normal_angle + np.sign(inc_angle - normal_angle) * phi_out
        m = np.tan(orient)
        b = y - m * x_int
        slopes.append(m)
        intercepts.append(b)
        kinks.append(x_int)
    return np.array(slopes), np.array(intercepts), np.array(kinks)


def compute_frame(
    t: float,
    n_ratio: float = params.ref_index_ratio,
    incoming_angles: Optional[Sequence[float]] = None,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return surface coordinates and ray parameters for a given ``t``."""
    x = surface_coordinates(t)
    slopes, intercepts, kinks = ray_parameters(t, n_ratio, incoming_angles)
    return x, slopes, intercepts, kinks
