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
from matplotlib.animation import FuncAnimation, PillowWriter
import os

from aberration import optics, params
from aberration.utils import build_patch, init_axes


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
        self.ext_lines: Lines = []
        self.surface = None
        self.focal_marker = None
        self.ext_focal_marker = None
        self.left_patch = None
        self.right_patch = None

    def _update(self, frame: int):
        t = self.t_values[frame]
        r = optics._surface_radius(t)
        x_surf = optics.surface_coordinates(t)
        self.surface.set_data(x_surf, params.surf_y)
        left_xy, right_xy = build_patch(x_surf)
        self.left_patch.set_xy(left_xy)
        self.right_patch.set_xy(right_xy)
        in_slopes = []
        in_intercepts = []
        for i, line in enumerate(self.lines):
            m_out = self.base_slopes[i]
            b_out = self.base_intercepts[i]
            x_int, y_int = _line_circle_intersection(m_out, b_out, r)
            normal_angle = np.arctan2(y_int, x_int - (params.plane_x - r))
            out_angle = np.arctan(m_out)
            in_angle = _incoming_angle(out_angle, normal_angle)
            m_in = np.tan(in_angle)
            b_in = y_int - m_in * x_int
            in_slopes.append(m_in)
            in_intercepts.append(b_in)
            y_start = m_in * params.x_start + b_in
            y_final = m_out * params.x_final + b_out
            line.set_data(
                [params.x_start, x_int, params.x_final],
                [y_start, y_int, y_final],
            )
            ext_line = self.ext_lines[i]
            y_ext = m_in * params.x_final + b_in
            ext_line.set_data(
                [x_int, params.x_final],
                [y_int, y_ext],
            )
        focal_x = _compute_focal_point(np.array(in_slopes), np.array(in_intercepts))
        self.ext_focal_marker.set_data([focal_x], [0.0])
        return (
            self.lines
            + self.ext_lines
            + [self.surface, self.left_patch, self.right_patch, self.focal_marker, self.ext_focal_marker]
        )

    def run(self, save: bool | None = None, prefix: str = "plane_wave") -> FuncAnimation:
        """Create and display the animation and optionally save images and a gif.

        If ``save`` is ``True`` the first and last frame are written as PNG files
        with ``prefix`` appended by ``_first.png`` and ``_last.png``. The full
        animation is also saved as ``prefix`` ``.gif``. Saving is automatically
        disabled when ``pytest`` is running.
        """
        fig, ax, self.left_patch, self.right_patch = init_axes()

        (self.surface,) = ax.plot([], [], lw=2, color="black")
        self.lines = [
            ax.plot([], [], color="red", lw=1)[0]
            for _ in range(params.n_rays)
        ]
        self.ext_lines = [
            ax.plot([], [], color="gray", lw=1)[0]
            for _ in range(params.n_rays)
        ]

        (self.focal_marker,) = ax.plot(
            self.focal_x,
            0.0,
            marker="o",
            color="red",
            markersize=8,
            lw=2,
            zorder=3,
        )

        # initial focal point for extended incoming rays
        r0 = optics._surface_radius(self.t_values[0])
        in_slopes = []
        in_intercepts = []
        for i in range(params.n_rays):
            m_out = self.base_slopes[i]
            b_out = self.base_intercepts[i]
            x_int, y_int = _line_circle_intersection(m_out, b_out, r0)
            normal_angle = np.arctan2(y_int, x_int - (params.plane_x - r0))
            out_angle = np.arctan(m_out)
            in_angle = _incoming_angle(out_angle, normal_angle)
            m_in = np.tan(in_angle)
            b_in = y_int - m_in * x_int
            in_slopes.append(m_in)
            in_intercepts.append(b_in)
        ext_focal_x = _compute_focal_point(np.array(in_slopes), np.array(in_intercepts))
        (self.ext_focal_marker,) = ax.plot(
            ext_focal_x,
            0.0,
            marker="o",
            color="gray",
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

        if save is None:
            save = "PYTEST_CURRENT_TEST" not in os.environ
        if save:
            self._update(0)
            fig.canvas.draw()
            fig.savefig(f"{prefix}_first.png")
            self._update(len(self.t_values) - 1)
            fig.canvas.draw()
            fig.savefig(f"{prefix}_last.png")
            writer = PillowWriter(fps=int(1000 / params.interval))
            anim.save(f"{prefix}.gif", writer=writer)

        plt.show()
        return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="Plane wave surface animation")
    parser.add_argument("--n-rays", type=int, default=params.n_rays, help="number of rays")
    parser.add_argument("--frames", type=int, default=params.frames, help="animation frames")
    parser.add_argument("--ray-range", type=float, default=params.y_range, help="half range of ray starting y positions")
    parser.add_argument("--plane-x", type=float, default=params.plane_x, help="x position of the plane at t=1")
    parser.add_argument("--radius", type=float, default=params.radius, help="initial surface radius")
    parser.add_argument("--far-radius", type=float, default=params.far_radius, help="effective radius at t=1")
    parser.add_argument("--aperture", type=float, default=params.aperture, help="half height of the optical element")
    parser.add_argument("--surf-samples", type=int, default=params.surf_samples, help="number of points for the surface")
    parser.add_argument("--interval", type=int, default=params.interval, help="animation frame interval (ms)")
    parser.add_argument("--n-ratio", type=float, default=params.ref_index_ratio, help="refractive index ratio")
    parser.add_argument("--x-start", type=float, default=params.x_start, help="x coordinate where rays start")
    parser.add_argument("--x-final", type=float, default=params.x_final, help="x coordinate where rays end")
    args = parser.parse_args()

    params.n_rays = args.n_rays
    params.frames = args.frames
    params.y_range = args.ray_range
    params.plane_x = args.plane_x
    params.radius = args.radius
    params.far_radius = args.far_radius
    params.aperture = args.aperture
    params.surf_samples = args.surf_samples
    params.interval = args.interval
    params.ref_index_ratio = args.n_ratio
    params.x_start = args.x_start
    params.x_final = args.x_final

    params.ys = np.linspace(-params.y_range, params.y_range, params.n_rays)
    params.surf_y = np.linspace(-params.aperture, params.aperture, params.surf_samples)
    params.ylim = (-params.aperture, params.aperture)
    params.xlim = (params.x_start, params.x_final)

    Animator().run()


if __name__ == "__main__":
    main()
