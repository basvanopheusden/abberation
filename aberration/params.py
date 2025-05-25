# Default parameters for the optical simulation

from __future__ import annotations

from typing import Tuple

import numpy as np

# number of rays and animation frames
n_rays: int = 10
frames: int = 120

# final incoming ray angles at ``t=1``
max_in_angle: float = 0.3

# vertical range for incoming rays
y_range: float = 0.4

# x position of the optical plane when the surface is flat
plane_x: float = 0.5

# radius of the initial spherical surface
radius: float = 0.5

# effective radius when ``t=1`` (approximate plane)
far_radius: float = 50.0

# half height of the optical element and vertical axis limits
aperture: float = 0.6

# number of samples used to draw the surface
surf_samples: int = 200

# interval between frames in milliseconds
interval: int = 50

# refractive index ratio used in ``compute_frame``
ref_index_ratio: float = 1.4

# x coordinates used for the rays
x_start: float = -1.0
x_final: float = 1.6

# default focal point where rays should converge
focal_point: Tuple[float, float] = (1.2, 0.0)

# axis limits and figure size
xlim = (-1.2, 1.7)
ylim = (-aperture, aperture)
figsize = (6, 4)

# phase offset for the background colors
phase0: float = 0.0

# Arrays derived from the parameters
incoming_final_angles = np.linspace(max_in_angle, -max_in_angle, n_rays)
ys = np.linspace(-y_range, y_range, n_rays)
surf_y = np.linspace(-aperture, aperture, surf_samples)
