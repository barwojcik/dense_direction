"""
Directional loss kernel functions.

This module contains a collection of functions that build kernels for directional loss.
"""

import numpy as np
import torch
from torch import Tensor
from mmengine import FUNCTIONS
from .utils import get_coordinates

__all__ = [
    "circular_point_kernel",
    "radial_line_kernel",
]


@FUNCTIONS.register_module()
def circular_point_kernel(
    pad: int = 2,
    div: int = 20,
    threshold: float = 1,
    **kwargs,
) -> Tensor:
    """
    Computes the circular point kernel.

    Returns a tensor that contains the circular kernels for directional loss computation.
    Each channel represents 2 points located on opposite sides of the circle (diameter = k_size).

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        threshold (float, optional): The distance threshold for points. Default: 1.0.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size).
    """
    assert threshold >= 1, "Distance threshold must be >= 1."
    k_size = 2 * pad + 1

    def compute_kernel(cords: np.ndarray, points: np.ndarray, threshold: float) -> np.ndarray:
        kernel = cords - points[:, :, np.newaxis, np.newaxis]
        kernel = threshold - np.sqrt((kernel**2).sum(0))
        kernel = np.where(kernel > 0, kernel, 0)
        kernel = kernel / kernel.sum(axis=(1, 2), keepdims=True)

        return kernel

    cords = get_coordinates(k_size)
    cords = np.stack(div * [cords], axis=1)

    angles = np.linspace(0, np.pi, div, False)
    half_points = np.stack([np.sin(angles), np.cos(angles)]) * (k_size - 1) / 2

    kernel = compute_kernel(cords, half_points, threshold)
    kernel += compute_kernel(cords, -half_points, threshold)
    kernel = kernel[..., ::-1]

    return torch.as_tensor(kernel / 2)


@FUNCTIONS.register_module()
def radial_line_kernel(
    pad: int = 2,
    div: int = 20,
    threshold: float = 1,
    **kwargs,
) -> Tensor:
    """
    This functions computes the radial line kernel.

    Returns a tensor that contains the radial line kernel for directional loss computation.
    Each channel represents a line that is a diameter of circle.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        threshold (float, optional): The distance threshold for lines. Default: 1.0.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """
    assert threshold >= 1, "Distance threshold must be >= 1."
    k_size = 2 * pad + 1

    cords = get_coordinates(k_size)
    cords = np.stack(div * [cords], axis=1)

    center_distances = np.sqrt((cords**2).sum(0))
    dists_weights = (pad - center_distances) / pad

    angles = np.linspace(0, np.pi, div, False)
    lines_params = np.stack([np.tan(angles), -np.ones_like(angles)])
    lines_params = lines_params[:, :, np.newaxis, np.newaxis]

    kernel = threshold - np.abs((cords * lines_params).sum(0)) / np.sqrt((lines_params**2).sum(0))
    kernel = np.where(kernel > 0, kernel, 0)
    kernel = np.where(center_distances > pad, 0, kernel)
    kernel = kernel * dists_weights
    kernel = kernel.transpose(0, 2, 1)
    kernel = kernel[:, :, ::-1]
    kernel = kernel / kernel.sum(axis=(-1, -2), keepdims=True)

    return torch.as_tensor(kernel)
