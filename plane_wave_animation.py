"""Animation of a plane wave interacting with a changing surface.

Scenario 1: A plane wave consisting of horizontal rays strikes a semi-circular
surface. The rays refract according to Snell's law and we compute an
approximate focal point for the outgoing bundle.

Scenario 2: The surface radius increases until it becomes nearly flat. The
outgoing portion of each ray remains fixed while the incoming side rotates to
continue satisfying Snell's law as the surface changes.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from aberration import optics, params


Lines = List[plt.Line2D]


def _compute_focal_point(slopes: np.ndarray, intercepts: np.ndarray) -> float:
    """Return the x location that best approximates the focal point."""
    slopes = np.asarray(slopes)
    intercepts = np.asarray(intercepts)
    return -np.sum(slopes * intercepts) / np.sum(slopes ** 2)


def _line_circle_intersection(m: float, b: float, r: float) -> tuple[float, float]:
    """Return intersection of ``y = m*x + b`` with the circle of radius ``r``."""
    xc = params.plane_x - r
    A = 1 + m**2
    B = 2 * (m * b - xc)
    C = xc**2 + b**2 - r**2
    disc = B**2 - 4 * A * C
    disc = max(disc, 0.0)
    x1 = (-B + np.sqrt(disc)) / (2 * A)
    x2 = (-B - np.sqrt(disc)) / (2 * A)
    # Choose the solution closest to ``plane_x``
    if abs(x1 - params.plane_x) < abs(x2 - params.plane_x):
        x = x1
    else:
        x = x2
    y = m * x + b
    return float(x), float(y)


def _incoming_angle(out_angle: float, normal_angle: float) -> float:
    """Return incoming angle required to produce ``out_angle``."""
    phi_out = abs(out_angle - normal_angle)
    phi_in = np.arcsin(
        np.clip(np.sin(phi_out) / params.ref_index_ratio, -1 + 1e-9, 1 - 1e-9)
    )
    return normal_angle + np.sign(out_angle - normal_angle) * phi_in


class Animator:
    def __init__(self):
        zero_angles = np.zeros(params.n_rays)
        x0, slopes, intercepts, _ = optics.compute_frame(0.0, incoming_angles=zero_angles)
        self.base_slopes = slopes
        self.base_intercepts = intercepts
        self.base_kinks = x0
        self.focal_x = _compute_focal_point(slopes, intercepts)
        self.t_values = np.linspace(0.0, 1.0, params.frames)
        self.lines: Lines = []
        self.surface = None
        self.focal_marker = None

    def _update(self, frame: int):
        t = self.t_values[frame]
        r = optics._surface_radius(t)
        x_surf = optics.surface_coordinates(t)
        self.surface.set_data(x_surf, params.surf_y)
        for i, line in enumerate(self.lines):
            m_out = self.base_slopes[i]
            b_out = self.base_intercepts[i]
            x_int, y_int = _line_circle_intersection(m_out, b_out, r)
            normal_angle = np.arctan2(y_int, x_int - (params.plane_x - r))
            out_angle = np.arctan(m_out)
            in_angle = _incoming_angle(out_angle, normal_angle)
            m_in = np.tan(in_angle)
            b_in = y_int - m_in * x_int
            y_start = m_in * params.x_start + b_in
            y_final = m_out * params.x_final + b_out
            line.set_data(
                [params.x_start, x_int, params.x_final],
                [y_start, y_int, y_final],
            )
        return self.lines + [self.surface, self.focal_marker]

    def run(self) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=params.figsize)
        ax.set_xlim(*params.xlim)
        ax.set_ylim(*params.ylim)
        ax.set_aspect("equal")
        ax.axis("off")

        (self.surface,) = ax.plot([], [], lw=2, color="black")
        self.lines = [ax.plot([], [], color="red")[0] for _ in range(params.n_rays)]

        (self.focal_marker,) = ax.plot(
            self.focal_x,
            0.0,
            marker="x",
            color="blue",
            markersize=8,
            lw=2,
            zorder=3,
        )

        anim = FuncAnimation(
            fig,
            self._update,
            frames=len(self.t_values),
            interval=params.interval,
            blit=True,
        )
        plt.show()
        return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="Plane wave surface animation")
    parser.parse_args()
    Animator().run()


if __name__ == "__main__":
    main()
