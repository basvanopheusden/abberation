import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


# Parameters
n_rays = 7
frames = 120

# y positions for incoming rays
ys = np.linspace(-0.4, 0.4, n_rays)

# Scenario 1: rays hit a circular lens and fail to meet at a single focus
scenario1 = []
for y in ys:
    start = [-1.0, y]
    lens_point = [0.0, y]
    # rays diverge slightly after passing through the lens
    final = [1.2, 0.2 * y]
    scenario1.append(np.array([start, lens_point, final]))
scenario1 = np.array(scenario1)

# Scenario 2: rays aim toward a focus but are refracted by a plane
plane_x = 0.5
scenario2 = []
for y in ys:
    start = [-1.0, y]
    # intersection with plane assuming ideal focus at (1, 0)
    slope = -y / 2.0
    y_plane = y + slope * (plane_x - (-1.0))
    plane_point = [plane_x, y_plane]
    # after refraction, rays miss the focus
    final = [1.2, 0.25 * y]
    scenario2.append(np.array([start, plane_point, final]))
scenario2 = np.array(scenario2)

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-1.2, 1.3)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect("equal")
ax.axis("off")

# background colors
left_patch = Polygon([], closed=True, fc="#F8F6ED", ec=None, zorder=0)
right_patch = Polygon([], closed=True, fc="#EFE9DE", ec=None, zorder=0)
ax.add_patch(left_patch)
ax.add_patch(right_patch)

# optical element that morphs from a semi-circle to a plane
radius = 0.5
# y-range is fixed so that we only see the segment between -0.6 and 0.6
surf_y = np.linspace(-0.6, 0.6, 200)
surface, = ax.plot([], [], lw=2, color="black")

# rays
lines = []
for _ in range(n_rays):
    line, = ax.plot([], [], color="red")
    lines.append(line)


def update(frame):
    phase = (frame / frames) * 2 * np.pi
    t = (np.sin(phase) + 1) / 2  # 0 -> 1 -> 0

    for i, line in enumerate(lines):
        p1 = scenario1[i]
        p2 = scenario2[i]
        pts = (1 - t) * p1 + t * p2
        line.set_data(pts[:, 0], pts[:, 1])

    # radius grows so that the semi-circle approaches a vertical plane
    start_angle = np.arcsin(np.clip(0.6 / radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(0.6 / 50)
    angle = (1 - t) * start_angle + t * end_angle
    r = 0.6 / np.sin(angle)
    x = np.where(
        np.abs(surf_y) <= r,
        np.sqrt(r**2 - surf_y**2) - r + plane_x,
        np.nan,
    )
    surface.set_data(x, surf_y)

    # update background patches so the color change follows the interface
    left_coords = np.vstack([
        [-1.2, surf_y[0]],
        [-1.2, surf_y[-1]],
        np.column_stack((x[::-1], surf_y[::-1])),
    ]).reshape(-1, 2)
    right_coords = np.vstack([
        np.column_stack((x, surf_y)),
        [1.3, surf_y[-1]],
        [1.3, surf_y[0]],
    ]).reshape(-1, 2)
    left_patch.set_xy(left_coords)
    right_patch.set_xy(right_coords)

    return lines + [surface, left_patch, right_patch]


ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

plt.show()
