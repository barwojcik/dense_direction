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
    "circular_point_kernels",
    "radial_line_kernels",
]


@FUNCTIONS.register_module()
def circular_point_kernels(
    pad: int = 2,
    div: int = 20,
    threshold:float = 1,
    **kwargs,
) -> Tensor:
    """
    Computes the circular point kernels.

    Returns a tensor that contains the circular kernels for directional loss computation.
    Each kernel represents 2 points located on opposite sides of the circle (diameter = k_size).

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        threshold (float, optional): The distance threshold for points. Default: 1.0.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size).
    """
    assert threshold >= 1, "Distance threshold must be >= 1."
    k_size = 2*pad+1

    def compute_kernels(cords: np.ndarray, points: np.ndarray, threshold:float) -> np.ndarray:
        kernels = cords - points[:, :, np.newaxis, np.newaxis]
        kernels = threshold - np.sqrt((kernels**2).sum(0))
        kernels = np.where(kernels > 0, kernels, 0)
        kernels = kernels / kernels.sum(axis=(1, 2), keepdims=True)

        return kernels

    cords = get_coordinates(k_size)
    cords = np.stack(div * [cords], axis=1)

    angles = np.linspace(0, np.pi, div, False)
    half_points = np.stack([np.sin(angles), np.cos(angles)]) * (k_size - 1) / 2

    kernels = compute_kernels(cords, half_points, threshold)
    kernels += compute_kernels(cords, -half_points, threshold)
    kernels = kernels[..., ::-1]

    return torch.as_tensor(kernels / 2)


@FUNCTIONS.register_module()
def radial_line_kernels(
    pad: int = 2,
    div: int = 20,
    threshold:float = 1,
    **kwargs,
) -> Tensor:
    """
    This functions computes the radial line kernels.

    Returns a tensor that contains the radial line kernels for directional loss computation.
    Each kernel represents the diameter of circle.

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

    kernels = threshold - np.abs((cords * lines_params).sum(0)) / np.sqrt((lines_params**2).sum(0))
    kernels = np.where(kernels > 0, kernels, 0)
    kernels = np.where(center_distances > pad, 0, kernels)
    kernels = kernels * dists_weights
    kernels = kernels.transpose(0, 2, 1)
    kernels = kernels[:, :, ::-1]
    kernels = kernels / kernels.sum(axis=(-1, -2), keepdims=True)

    return torch.as_tensor(kernels)
