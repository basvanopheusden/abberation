import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


# Parameters
n_rays = 7
frames = 120

# y positions for incoming rays
ys = np.linspace(-0.4, 0.4, n_rays)

plane_x = 0.5

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-1.2, 1.3)
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
        [1.3, -0.6],
        [1.3, 0.6],
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


def compute_frame(t, n_ratio=1.4):
    """Return surface coordinates and ray parameters for a given ``t``.

    Parameters
    ----------
    t : float
        Interpolation parameter between ``0`` and ``1``.

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

    for y in ys:
        if np.abs(y) <= r:
            x_int = np.sqrt(r**2 - y**2) - r + plane_x
            normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        else:
            x_int = np.nan
            normal_angle = 0.0

        phi_in = np.abs(normal_angle)
        phi_out = np.arcsin(np.sin(phi_in) / n_ratio)
        orient = normal_angle - np.sign(normal_angle) * phi_out
        m = np.tan(orient)
        b = y - m * x_int

        slopes.append(m)
        intercepts.append(b)
        kinks.append(x_int)

    return x, np.array(slopes), np.array(intercepts), np.array(kinks)


def update(frame):
    phase = (frame / frames) * 2 * np.pi
    t = (np.sin(phase) + 1) / 2  # 0 -> 1 -> 0

    x, slopes, intercepts, kinks = compute_frame(t)

    for i, line in enumerate(lines):
        x_int = kinks[i]
        y_int = ys[i]
        x_final = 1.2
        y_final = slopes[i] * x_final + intercepts[i]
        line.set_data([-1.0, x_int, x_final], [ys[i], y_int, y_final])

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
            [1.3, -0.6],
            [1.3, 0.6],
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
