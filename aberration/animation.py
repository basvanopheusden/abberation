"""Matplotlib animation helpers."""

from __future__ import annotations

from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from scipy.signal import medfilt

from . import optics, params
from .analysis import find_optimal_max_in_angle
from .utils import build_patch, init_axes


Lines = List[plt.Line2D]


# mutable objects used while animating
fig = None
ax = None
lines: Lines = []
surface = None
left_patch = None
right_patch = None
focal_marker = None
optimal_angles: List[float] = []
optimal_distances: List[float] = []
t_values: List[float] = []


def _calculate_optimal_angles() -> tuple[list[float], list[float], list[float]]:
    """Return ``t`` values and optimal angles/distances for each frame."""
    t_vals: list[float] = []
    angles: list[float] = []
    dists: list[float] = []
    for frame in range(params.frames):
        phase = (frame / params.frames) * 2 * np.pi
        t = (np.sin(phase) + 1) / 2
        t_vals.append(t)
        angle, dist = find_optimal_max_in_angle(t, focal_point=params.focal_point)
        angles.append(angle)
        dists.append(dist)
    return t_vals, angles, dists


def _smooth_angles(t_vals: list[float], angles: list[float]) -> list[float]:
    """Return ``angles`` after applying a median filter."""
    sort_idx = np.argsort(t_vals)
    unsort_idx = np.argsort(sort_idx)
    angles_sorted = np.array(angles)[sort_idx]
    if angles_sorted.size >= 5:
        angles_sorted = medfilt(angles_sorted, kernel_size=5)
    return list(angles_sorted[unsort_idx])


def _plot_diagnostics(t_vals: list[float], angles: list[float], dists: list[float]) -> None:
    """Plot optimization diagnostics before running the animation."""
    sort_idx = np.argsort(t_vals)
    t_sorted = np.array(t_vals)[sort_idx]
    angles_sorted = np.array(angles)[sort_idx]
    dists_sorted = np.array(dists)[sort_idx]
    plt.figure()
    plt.plot(t_sorted, angles_sorted)
    plt.xlabel("t")
    plt.ylabel("optimal angle (rad)")
    plt.title("Optimal angle vs t")
    plt.show()
    plt.figure()
    plt.plot(t_sorted, dists_sorted)
    plt.xlabel("t")
    plt.ylabel("distance")
    plt.title("Minimum distance vs t")
    plt.show()


def update(frame: int):
    """Animation callback updating all artists for ``frame``."""
    t = t_values[frame]
    best_angle = optimal_angles[frame]
    in_angles = np.linspace(best_angle, -best_angle, params.n_rays)
    x, slopes, intercepts, kinks = optics.compute_frame(
        t, n_ratio=params.ref_index_ratio, incoming_angles=in_angles
    )
    for i, line in enumerate(lines):
        x_int = kinks[i]
        y_int = params.ys[i]
        in_angle = t * in_angles[i]
        m_in = np.tan(in_angle)
        y_start = y_int - m_in * (x_int - params.x_start)
        y_final = slopes[i] * params.x_final + intercepts[i]
        line.set_data(
            [params.x_start, x_int, params.x_final], [y_start, y_int, y_final]
        )
    surface.set_data(x, params.surf_y)
    left_xy, right_xy = build_patch(x)
    left_patch.set_xy(left_xy)
    right_patch.set_xy(right_xy)
    return lines + [surface, left_patch, right_patch, focal_marker]


def run_animation(
    save: bool | None = None, prefix: str = "aberration"
) -> FuncAnimation:
    """Create and display the matplotlib animation and return it.

    If ``save`` is ``True`` the first and last frame are written as PNG files
    with ``prefix`` appended by ``_first.png`` and ``_last.png``.  The complete
    animation is also stored as ``prefix`` ``.gif``.  Saving is automatically
    disabled when ``pytest`` is running.
    """
    global fig, ax, lines, surface, left_patch, right_patch, focal_marker, optimal_angles, optimal_distances, t_values

    print("Calculating optimal angles ...")
    t_values, optimal_angles, optimal_distances = _calculate_optimal_angles()
    print("Finished calculating optimal angles.")

    optimal_angles = _smooth_angles(t_values, optimal_angles)

    _plot_diagnostics(t_values, optimal_angles, optimal_distances)
    for t_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        angle, _ = find_optimal_max_in_angle(t_val, focal_point=params.focal_point)
        print(f"t={t_val:.1f} optimal angle={angle:.3f}")

    fig, ax, left_patch, right_patch = init_axes()

    # draw the focal point as a blue cross
    global focal_marker
    (focal_marker,) = ax.plot(
        params.focal_point[0],
        params.focal_point[1],
        marker="x",
        color="blue",
        markersize=8,
        lw=2,
        zorder=3,
    )

    (surface,) = ax.plot([], [], lw=2, color="black")

    lines = []
    for _ in range(params.n_rays):
        (line,) = ax.plot([], [], color="red")
        lines.append(line)

    # keep a reference to the animation object so it is not garbage collected
    anim = FuncAnimation(
        fig, update, frames=params.frames, interval=params.interval, blit=True
    )

    if save is None:
        save = "PYTEST_CURRENT_TEST" not in os.environ
    if save:
        update(0)
        fig.canvas.draw()
        fig.savefig(f"{prefix}_first.png")
        update(len(t_values) - 1)
        fig.canvas.draw()
        fig.savefig(f"{prefix}_last.png")
        writer = PillowWriter(fps=int(1000 / params.interval))
        anim.save(f"{prefix}.gif", writer=writer)

    plt.show()
    return anim
