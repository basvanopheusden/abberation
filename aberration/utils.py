"""Utility helpers for the aberration animations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from . import params, optics


def build_patch(x_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordinates for the background patches.

    The patches fill the regions to the left and right of the optical
    surface so that the background color matches the interface.
    """
    mask = ~np.isnan(x_values)
    left_xy = (
        [params.xlim[0], params.ylim[0]],
        [params.xlim[0], params.ylim[1]],
        *zip(x_values[mask][::-1], params.surf_y[mask][::-1]),
    )
    right_xy = (
        [params.xlim[1], params.ylim[0]],
        [params.xlim[1], params.ylim[1]],
        *zip(x_values[mask][::-1], params.surf_y[mask][::-1]),
    )
    return np.array(left_xy), np.array(right_xy)


def init_axes() -> Tuple[plt.Figure, plt.Axes, Polygon, Polygon]:
    """Return a matplotlib figure, axis and background patches."""
    fig, ax = plt.subplots(figsize=params.figsize)
    ax.set_xlim(*params.xlim)
    ax.set_ylim(*params.ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    t0 = (np.sin(params.phase0) + 1) / 2
    x0 = optics.surface_coordinates(t0)
    left_xy, right_xy = build_patch(x0)
    left_patch = Polygon(left_xy, closed=True, fc="#F8F6ED", ec=None, zorder=0)
    right_patch = Polygon(right_xy, closed=True, fc="#EFE9DE", ec=None, zorder=0)
    ax.add_patch(left_patch)
    ax.add_patch(right_patch)

    return fig, ax, left_patch, right_patch
