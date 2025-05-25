"""Optical aberration simulation package."""

from . import params
from .analysis import distance_to_focus, find_optimal_max_in_angle, total_distance_to_focus
from .optics import compute_frame, ray_parameters, surface_coordinates
from .utils import build_patch, init_axes

__all__ = [
    "compute_frame",
    "ray_parameters",
    "surface_coordinates",
    "distance_to_focus",
    "total_distance_to_focus",
    "find_optimal_max_in_angle",
    "build_patch",
    "init_axes",
    "params",
]

