import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


# Parameters
# Number of rays and animation frames
n_rays = 11
frames = 120

# final incoming ray angles at ``t=1``.  The rays start out horizontal and
# smoothly rotate to these angles.
# final incoming ray angles. Rays above the horizontal should point
# downward and those below should point upward when ``t=1``.  This is
# accomplished by reversing the sign of the equally spaced angles so they
# run from positive to negative as ``ys`` increases.
max_in_angle = 0.3
incoming_final_angles = np.linspace(max_in_angle, -max_in_angle, n_rays)

# y positions for incoming rays
ys = np.linspace(-0.4, 0.4, n_rays)

plane_x = 0.5

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-1.2, 1.7)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect("equal")
ax.axis("off")

radius = 0.5
# y-range is fixed so that we only see the segment between -0.6 and 0.6
surf_y = np.linspace(-0.6, 0.6, 200)

# background colors following the optical interface
phase0 = 0
t0 = (np.sin(phase0) + 1) / 2
start_angle = np.arcsin(np.clip(0.6 / radius, -1 + 1e-9, 1 - 1e-9))
end_angle = np.arcsin(0.6 / 50)
angle0 = (1 - t0) * start_angle + t0 * end_angle
r0 = 0.6 / np.sin(angle0)
x0 = np.where(
    np.abs(surf_y) <= r0,
    np.sqrt(r0**2 - surf_y**2) - r0 + plane_x,
    np.nan,
)
mask0 = ~np.isnan(x0)
left_xy = np.vstack(
    (
        [-1.2, -0.6],
        [-1.2, 0.6],
        np.column_stack((x0[mask0][::-1], surf_y[mask0][::-1])),
    )
)
right_xy = np.vstack(
    (
        [1.7, -0.6],
        [1.7, 0.6],
        # traverse the optical surface from top to bottom so that the
        # patch boundary exactly follows the semicircle
        np.column_stack((x0[mask0][::-1], surf_y[mask0][::-1])),
    )
)
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
    start_angle = np.arcsin(np.clip(0.6 / radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(0.6 / 50)
    angle = (1 - t) * start_angle + t * end_angle
    r = 0.6 / np.sin(angle)
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

    x, slopes, intercepts, kinks = compute_frame(t)

    for i, line in enumerate(lines):
        x_int = kinks[i]
        y_int = ys[i]

        # starting point of the incoming ray depends on its angle
        in_angle = t * incoming_final_angles[i]
        m_in = np.tan(in_angle)
        y_start = y_int - m_in * (x_int + 1.0)

        x_final = 1.6
        y_final = slopes[i] * x_final + intercepts[i]
        line.set_data([-1.0, x_int, x_final], [y_start, y_int, y_final])

    surface.set_data(x, surf_y)
    mask = ~np.isnan(x)
    left_xy = np.vstack(
        (
            [-1.2, -0.6],
            [-1.2, 0.6],
            np.column_stack((x[mask][::-1], surf_y[mask][::-1])),
        )
    )
    right_xy = np.vstack(
        (
            [1.7, -0.6],
            [1.7, 0.6],
            # use the same orientation as ``left_xy`` so the interface
            # between the two patches precisely matches the optical surface
            np.column_stack((x[mask][::-1], surf_y[mask][::-1])),
        )
    )
    left_patch.set_xy(left_xy)
    right_patch.set_xy(right_xy)
    return lines + [surface, left_patch, right_patch]


ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

plt.show()
