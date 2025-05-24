import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------
# Configurable parameters with default values. These values are overwritten
# when the module is executed as a script and command line arguments are
# provided.  They remain unchanged when imported so that the tests continue
# to use the defaults.
# ---------------------------------------------------------------------------

# number of rays and animation frames
n_rays = 11
frames = 120

# final incoming ray angles at ``t=1``.  The rays start out horizontal and
# smoothly rotate to these angles. Rays above the horizontal should point
# downward and those below should point upward when ``t=1``.  This is
# accomplished by reversing the sign of the equally spaced angles so they
# run from positive to negative as ``ys`` increases.
max_in_angle = 0.3

# vertical range for incoming rays
y_range = 0.4

# x position of the optical plane when the surface is flat
plane_x = 0.5

# radius of the initial spherical surface
radius = 0.5

# effective radius when ``t=1`` (approximate plane)
far_radius = 50.0

# half height of the optical element and vertical axis limits
aperture = 0.6

# number of samples used to draw the surface
surf_samples = 200

# interval between frames in milliseconds
interval = 50

# refractive index ratio used in ``compute_frame``
ref_index_ratio = 1.4

# x coordinates used for the rays
x_start = -1.0
x_final = 1.6

# axis limits and figure size
xlim = (-1.2, 1.7)
ylim = (-aperture, aperture)
figsize = (6, 4)

# phase offset for the background colors
phase0 = 0.0


# Arrays derived from the parameters. These will be updated if command line
# arguments are supplied.
incoming_final_angles = np.linspace(max_in_angle, -max_in_angle, n_rays)
ys = np.linspace(-y_range, y_range, n_rays)
surf_y = np.linspace(-aperture, aperture, surf_samples)


# placeholders filled when the animation is created
fig = None
ax = None
lines = []
surface = None
left_patch = None
right_patch = None


def build_patch(x_values):
    """Return coordinates for the background patches.

    Parameters
    ----------
    x_values : np.ndarray
        x coordinates of the optical surface corresponding to ``surf_y``.

    Returns
    -------
    tuple of np.ndarray
        ``(left_xy, right_xy)`` polygon vertices for the left and right
        background patches.
    """

    mask = ~np.isnan(x_values)
    left_xy = np.vstack(
        (
            [xlim[0], ylim[0]],
            [xlim[0], ylim[1]],
            np.column_stack((x_values[mask][::-1], surf_y[mask][::-1])),
        )
    )
    right_xy = np.vstack(
        (
            [xlim[1], ylim[0]],
            [xlim[1], ylim[1]],
            np.column_stack((x_values[mask][::-1], surf_y[mask][::-1])),
        )
    )
    return left_xy, right_xy


# ---------------------------------------------------------------------------
# Runtime setup is deferred until ``run_animation`` is called.  This keeps the
# module importable without creating figures, which is important for the tests.
# ---------------------------------------------------------------------------


def compute_frame(t, n_ratio=1.4, incoming_angles=None):
    """Return surface coordinates and ray parameters for a given ``t``.

    Parameters
    ----------
    t : float
        Interpolation parameter between ``0`` and ``1``.
    n_ratio : float, optional
        Ratio of refractive indices used when applying Snell's law.
    incoming_angles : array-like of float, optional
        Final incoming angles for each ray. When ``None`` the module level
        ``incoming_final_angles`` are used.

    Returns
    -------
    tuple
        ``(x, slopes, intercepts, kinks)`` where ``x`` are the x coordinates of
        the optical surface corresponding to ``surf_y``. ``slopes`` and
        ``intercepts`` describe the refracted rays and ``kinks`` gives the x
        position where each incoming ray meets the surface.
    """
    start_angle = np.arcsin(np.clip(aperture / radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(aperture / far_radius)
    angle = (1 - t) * start_angle + t * end_angle
    r = aperture / np.sin(angle)
    x = np.where(
        np.abs(surf_y) <= r,
        np.sqrt(r**2 - surf_y**2) - r + plane_x,
        np.nan,
    )

    slopes = []
    intercepts = []
    kinks = []

    if incoming_angles is None:
        incoming_angles = incoming_final_angles

    for idx, y in enumerate(ys):
        if np.abs(y) <= r:
            x_int = np.sqrt(r**2 - y**2) - r + plane_x
            normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        else:
            x_int = np.nan
            normal_angle = 0.0

        # orientation of the incoming ray for this ``t``
        inc_angle = t * incoming_angles[idx]

        phi_in = np.abs(inc_angle - normal_angle)
        # multiply by ``n_ratio`` so that ``n_ratio > 1`` focuses the rays
        phi_out = np.arcsin(
            np.clip(np.sin(phi_in) * n_ratio, -1 + 1e-9, 1 - 1e-9)
        )
        orient = normal_angle + np.sign(inc_angle - normal_angle) * phi_out
        m = np.tan(orient)
        b = y - m * x_int

        slopes.append(m)
        intercepts.append(b)
        kinks.append(x_int)

    return x, np.array(slopes), np.array(intercepts), np.array(kinks)


def distance_to_focus(slope, intercept, focal_point=(1.2, 0.0)):
    """Return the perpendicular distance from a ray to a focal point.

    The candidate ray is represented by ``y = slope * x + intercept``. The
    function computes the shortest distance from this line to ``focal_point``.

    Parameters
    ----------
    slope : float or ``np.ndarray``
        The slope(s) of the candidate ray(s).
    intercept : float or ``np.ndarray``
        The intercept(s) of the candidate ray(s).
    focal_point : tuple of float, optional
        The ``(x, y)`` coordinates of the focal point. Defaults to ``(1.2, 0)``.

    Returns
    -------
    float or ``np.ndarray``
        The perpendicular distance(s) from the ray(s) to ``focal_point``.
    """

    x0, y0 = focal_point
    slope = np.asarray(slope)
    intercept = np.asarray(intercept)
    return np.abs(slope * x0 + intercept - y0) / np.sqrt(slope**2 + 1)


def total_distance_to_focus(t, max_in_angle, focal_point=(1.2, 0.0)):
    """Return the summed distance of all rays to ``focal_point``.

    Parameters
    ----------
    t : float
        Interpolation parameter passed to :func:`compute_frame`.
    max_in_angle : float
        Maximum incoming angle used to generate the rays.  The individual
        incoming angles are spaced linearly from ``max_in_angle`` to
        ``-max_in_angle`` while keeping their vertical intercepts fixed.
    focal_point : tuple of float, optional
        Coordinates of the focal point.

    Returns
    -------
    float
        Sum of :func:`distance_to_focus` for all rays.
    """

    in_angles = np.linspace(max_in_angle, -max_in_angle, n_rays)
    _, slopes, intercepts, _ = compute_frame(t, incoming_angles=in_angles)
    return float(np.sum(distance_to_focus(slopes, intercepts, focal_point)))


def find_optimal_max_in_angle(t, focal_point=(1.2, 0.0), search_angles=None):
    """Return the ``max_in_angle`` that minimizes the total distance.

    Parameters
    ----------
    t : float
        Interpolation parameter passed to :func:`compute_frame`.
    focal_point : tuple of float, optional
        Coordinates of the focal point.
    search_angles : array-like of float, optional
        Sequence of candidate ``max_in_angle`` values to search.  When ``None``
        a default grid between ``0`` and ``0.6`` radians is used.

    Returns
    -------
    tuple
        ``(best_angle, min_total_distance)`` where ``best_angle`` is the angle
        from ``search_angles`` that yields the smallest total distance and
        ``min_total_distance`` is that minimal distance.
    """

    if search_angles is None:
        search_angles = np.linspace(0.0, 0.6, 50)

    best_angle = None
    min_dist = np.inf

    for angle in search_angles:
        dist = total_distance_to_focus(t, angle, focal_point)
        if dist < min_dist:
            min_dist = dist
            best_angle = angle

    return best_angle, min_dist


def update(frame):
    phase = (frame / frames) * 2 * np.pi
    t = (np.sin(phase) + 1) / 2  # 0 -> 1 -> 0

    x, slopes, intercepts, kinks = compute_frame(t, n_ratio=ref_index_ratio)

    for i, line in enumerate(lines):
        x_int = kinks[i]
        y_int = ys[i]

        # starting point of the incoming ray depends on its angle
        in_angle = t * incoming_final_angles[i]
        m_in = np.tan(in_angle)
        y_start = y_int - m_in * (x_int - x_start)

        y_final = slopes[i] * x_final + intercepts[i]
        line.set_data([x_start, x_int, x_final], [y_start, y_int, y_final])

    surface.set_data(x, surf_y)
    left_xy, right_xy = build_patch(x)
    left_patch.set_xy(left_xy)
    right_patch.set_xy(right_xy)
    return lines + [surface, left_patch, right_patch]


def run_animation():
    """Create the matplotlib animation using the current global parameters."""
    global fig, ax, lines, surface, left_patch, right_patch

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    # initial background following the optical interface
    t0 = (np.sin(phase0) + 1) / 2
    start_angle = np.arcsin(np.clip(aperture / radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(aperture / far_radius)
    angle0 = (1 - t0) * start_angle + t0 * end_angle
    r0 = aperture / np.sin(angle0)
    x0 = np.where(
        np.abs(surf_y) <= r0,
        np.sqrt(r0**2 - surf_y**2) - r0 + plane_x,
        np.nan,
    )
    left_xy, right_xy = build_patch(x0)
    left_patch = Polygon(left_xy, closed=True, fc="#F8F6ED", ec=None, zorder=0)
    right_patch = Polygon(right_xy, closed=True, fc="#EFE9DE", ec=None, zorder=0)
    ax.add_patch(left_patch)
    ax.add_patch(right_patch)

    # optical element that morphs from a semi-circle to a plane
    surface, = ax.plot([], [], lw=2, color="black")

    # rays
    lines = []
    for _ in range(n_rays):
        line, = ax.plot([], [], color="red")
        lines.append(line)

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical aberration animation")
    parser.add_argument("--n-rays", type=int, default=n_rays, help="number of rays")
    parser.add_argument("--frames", type=int, default=frames, help="animation frames")
    parser.add_argument("--max-in-angle", type=float, default=max_in_angle, help="maximum incoming angle (radians)")
    parser.add_argument("--ray-range", type=float, default=y_range, help="half range of ray starting y positions")
    parser.add_argument("--plane-x", type=float, default=plane_x, help="x position of the plane at t=1")
    parser.add_argument("--radius", type=float, default=radius, help="initial surface radius")
    parser.add_argument("--far-radius", type=float, default=far_radius, help="effective radius at t=1")
    parser.add_argument("--aperture", type=float, default=aperture, help="half height of the optical element")
    parser.add_argument("--surf-samples", type=int, default=surf_samples, help="number of points for the surface")
    parser.add_argument("--interval", type=int, default=interval, help="animation frame interval (ms)")
    parser.add_argument("--n-ratio", type=float, default=ref_index_ratio, help="refractive index ratio")
    parser.add_argument("--x-start", type=float, default=x_start, help="x coordinate where rays start")
    parser.add_argument("--x-final", type=float, default=x_final, help="x coordinate where rays end")
    args = parser.parse_args()

    n_rays = args.n_rays
    frames = args.frames
    max_in_angle = args.max_in_angle
    y_range = args.ray_range
    plane_x = args.plane_x
    radius = args.radius
    far_radius = args.far_radius
    aperture = args.aperture
    surf_samples = args.surf_samples
    interval = args.interval
    ref_index_ratio = args.n_ratio
    x_start = args.x_start
    x_final = args.x_final

    incoming_final_angles = np.linspace(max_in_angle, -max_in_angle, n_rays)
    ys = np.linspace(-y_range, y_range, n_rays)
    surf_y = np.linspace(-aperture, aperture, surf_samples)
    ylim = (-aperture, aperture)

    run_animation()
