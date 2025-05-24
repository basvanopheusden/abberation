"""Command line entry point for the optical aberration animation."""

from __future__ import annotations

import argparse
import numpy as np

from aberration import animation, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical aberration animation")
    parser.add_argument("--n-rays", type=int, default=params.n_rays, help="number of rays")
    parser.add_argument("--frames", type=int, default=params.frames, help="animation frames")
    parser.add_argument("--max-in-angle", type=float, default=params.max_in_angle, help="maximum incoming angle (radians)")
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
    parser.add_argument(
        "--focal-point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=params.focal_point,
        help="focal point coordinates as 'x y'",
    )
    args = parser.parse_args()

    params.n_rays = args.n_rays
    params.frames = args.frames
    params.max_in_angle = args.max_in_angle
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
    params.focal_point = tuple(args.focal_point)

    params.incoming_final_angles = np.linspace(params.max_in_angle, -params.max_in_angle, params.n_rays)
    params.ys = np.linspace(-params.y_range, params.y_range, params.n_rays)
    params.surf_y = np.linspace(-params.aperture, params.aperture, params.surf_samples)
    params.ylim = (-params.aperture, params.aperture)

    animation.run_animation()


if __name__ == "__main__":
    main()