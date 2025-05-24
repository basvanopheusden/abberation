import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
final_x = 1.2
n_ratio = 1.5  # n1 / n2 for Snell's law
scenario2 = []
for y in ys:
    start = [-1.0, y]
    # intersection with plane assuming ideal focus at (1, 0)
    slope_in = -y / 2.0
    y_plane = y + slope_in * (plane_x - (-1.0))
    plane_point = [plane_x, y_plane]

    # apply Snell's law to determine outgoing angle
    sin_in = abs(slope_in) / np.sqrt(1 + slope_in**2)
    sin_out = sin_in / n_ratio
    slope_out = np.sign(slope_in) * sin_out / np.sqrt(1 - sin_out**2)

    final_y = y_plane + slope_out * (final_x - plane_x)
    final = [final_x, final_y]
    scenario2.append(np.array([start, plane_point, final]))
scenario2 = np.array(scenario2)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-1.2, 1.3)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # optical element that morphs from a semi-circle to a plane
    radius = 0.5
    # y-range is fixed so that we only see the segment between -0.6 and 0.6
    surf_y = np.linspace(-0.6, 0.6, 200)
    surface, = ax.plot([], [], lw=2, color="blue")

    # rays
    lines = []
    for _ in range(n_rays):
        line, = ax.plot([], [], color="orange")
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
        return lines + [surface]

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.show()
