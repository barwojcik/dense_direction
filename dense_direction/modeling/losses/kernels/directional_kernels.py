"""
Directional loss kernel functions.

This module contains a collection of functions that build kernels for directional loss.
"""

import numpy as np
import torch
from torch import Tensor
from mmengine import FUNCTIONS
from .utils import (
    get_kernel_size,
    get_stacked_coordinates,
    get_disc_mask,
    get_points_on_semicircle,
)

__all__ = [
    "circular_point_kernel",
    "radial_line_kernel",
    "polar_kernel",
    "polar_disc_kernel",
    "polar_wedge_kernel",
    "polar_sector_kernel",
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

    k_size: int = get_kernel_size(pad)

    def compute_kernel(cords: np.ndarray, points: np.ndarray, threshold: float) -> np.ndarray:
        kernel = cords - points[:, :, np.newaxis, np.newaxis]
        kernel = threshold - np.sqrt((kernel**2).sum(0))
        kernel = np.where(kernel > 0, kernel, 0)
        kernel = kernel / kernel.sum(axis=(1, 2), keepdims=True)

        return kernel

    stacked_cords: np.ndarray = get_stacked_coordinates(k_size, div)

    half_points: np.ndarray = get_points_on_semicircle(div) * (k_size - 1) / 2

    kernel: np.ndarray = compute_kernel(stacked_cords, half_points, threshold)
    kernel += compute_kernel(stacked_cords, -half_points, threshold)
    kernel = kernel[..., ::-1]

    return torch.as_tensor(kernel / 2)


@FUNCTIONS.register_module()
def radial_line_kernel(
    pad: int = 2,
    div: int = 20,
    threshold: float = 1,
    reverse_weights: bool = False,
    **kwargs,
) -> Tensor:
    """
    This functions computes the radial line kernel.

    Returns a tensor that contains the radial line kernel for directional loss computation.
    Each channel represents a line that is a diameter of circle.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        threshold (float, optional): The distance threshold from the line. Default: 1.0.
        reverse_weights (bool, optional): Whether to reverse the distance weights. Default: False.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """

    assert threshold >= 1, "Distance threshold must be >= 1."

    k_size: int = get_kernel_size(pad)

    stacked_cords: np.ndarray = get_stacked_coordinates(k_size, div)

    center_distances: np.ndarray = np.sqrt((stacked_cords**2).sum(0))
    dists_weights: np.ndarray = (pad - center_distances) / pad

    if reverse_weights:
        dists_weights = 1 - dists_weights

    angles: np.ndarray = np.linspace(0, np.pi, div, False)
    lines_params: np.ndarray = np.stack([np.tan(angles), -np.ones_like(angles)])
    lines_params = lines_params[:, :, np.newaxis, np.newaxis]

    kernel: np.ndarray = threshold - np.abs((stacked_cords * lines_params).sum(0)) / np.sqrt((lines_params**2).sum(0))
    kernel = np.where(kernel > 0, kernel, 0)
    kernel = np.where(center_distances > pad, 0, kernel)
    kernel = kernel * dists_weights
    kernel = kernel.transpose(0, 2, 1)
    kernel = kernel[:, :, ::-1]
    kernel = kernel / kernel.sum(axis=(-1, -2), keepdims=True)

    return torch.as_tensor(kernel)


@FUNCTIONS.register_module()
def polar_kernel(
    pad: int = 2,
    div: int = 20,
    alpha: float = 1,
    **kwargs,
) -> Tensor:
    """
    This functions computes the polar kernel.

    Returns a tensor that contains the polar kernel for directional loss computation.
    Each channel represents a normalized angular distance from a different direction.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        alpha (float, optional): The exponent parameter for the polar kernel. Default: 1.0.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """

    k_size: int = get_kernel_size(pad)

    stacked_cords: np.ndarray = get_stacked_coordinates(k_size, div)
    stacked_cords = stacked_cords/np.linalg.norm(stacked_cords, axis=0, keepdims=True)

    points: np.ndarray = get_points_on_semicircle(div)

    kernel: np.ndarray = np.arccos(np.einsum("ijkl,ij->jkl", stacked_cords, points))
    kernel = np.where(kernel > .5*np.pi, np.pi-kernel, kernel)
    kernel /= .5*np.pi
    kernel = (1 - kernel)**alpha
    kernel = np.nan_to_num(kernel)
    kernel = kernel[:, :, ::-1]
    kernel = kernel / kernel.sum(axis=(-1, -2), keepdims=True)

    return torch.as_tensor(kernel)


@FUNCTIONS.register_module()
def polar_wedge_kernel(
    pad: int = 2,
    div: int = 20,
    alpha: float = 1,
    threshold: float = np.pi/10,
    **kwargs,
) -> Tensor:
    """
    This functions computes the polar wedge kernel.

    Returns a tensor that contains the polar wedge kernel for directional loss computation.
    Each channel represents a normalized angular distance from a different direction limited by a
    threshold, forming symmetrical wedges.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        alpha (float, optional): The exponent parameter for the polar kernel. Default: 1.0.
        threshold (float, optional): The angular distance threshold for wedges in radians.
            It Should be in range from 0 to 0.25π. Default: 0.1π.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """

    assert 0 < threshold <= 0.25 * np.pi, \
        "Angular distance threshold must be in range from 0 to 0.25π."

    k_size: int = get_kernel_size(pad)

    stacked_cords: np.ndarray = get_stacked_coordinates(k_size, div)
    stacked_cords = stacked_cords / np.linalg.norm(stacked_cords, axis=0, keepdims=True)

    points: np.ndarray = get_points_on_semicircle(div)

    kernel: np.ndarray = np.arccos(np.einsum("ijkl,ij->jkl", stacked_cords, points))
    kernel = np.where(kernel > .5*np.pi, np.pi-kernel, kernel)
    kernel = threshold - kernel
    kernel = np.where(kernel < 0, 0, kernel)
    kernel /= threshold
    kernel = kernel**alpha
    kernel = np.nan_to_num(kernel)
    kernel = kernel[:, :, ::-1]
    kernel = kernel / kernel.sum(axis=(-1, -2), keepdims=True)

    return torch.as_tensor(kernel)


@FUNCTIONS.register_module()
def polar_disc_kernel(
    pad: int = 2,
    div: int = 20,
    alpha: float = 1,
    radius: float=None,
    **kwargs,
) -> Tensor:
    """
    This functions computes the polar disc kernel.

    Returns a tensor that contains the polar disc kernel for directional loss computation.
    Each channel represents a normalized angular distance from a different direction masked by a
    disc, forming symmetrical wedges.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        alpha (float, optional): The exponent parameter for the polar kernel. Default: 1.0.
        radius (float, optional): Radius parameter for a disc mask. Default: pad.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """
    k_size: int = get_kernel_size(pad)
    disc_mask: np.ndarray = get_disc_mask(k_size, radius)

    kernel: Tensor = polar_kernel(pad, div, alpha)
    kernel = kernel * disc_mask
    kernel /= kernel.sum(dim=(1, 2), keepdim=True)

    return kernel


@FUNCTIONS.register_module()
def polar_sector_kernel(
    pad: int = 2,
    div: int = 20,
    alpha: float = 1,
    threshold: float = np.pi/10,
    radius: float=None,
    **kwargs,
) -> Tensor:
    """
    This functions computes the polar sector kernel.

    Returns a tensor that contains the polar sector kernel for directional loss computation.
    Each channel represents a normalized angular distance from a different direction limited by a
    threshold and masked by a disc, forming symmetrical disc sectors.

    Args:
        pad (int): Padding value for a kernel. Default: 2.
        div (int): Division factor for direction bins. Default: 20.
        alpha (float, optional): The exponent parameter for the polar kernel. Default: 1.0.
        threshold (float, optional): The angular distance threshold for wedges in radians.
            It Should be in range from 0 to 0.25π. Default: 0.1π.
        radius (float, optional): Radius parameter for a disc mask. Default: pad.

    Returns:
        Tensor: Tensor of shape (div, k_size, k_size), where k_size = 2 * pad +1.
    """
    k_size: int = get_kernel_size(pad)
    disc_mask: np.ndarray = get_disc_mask(k_size, radius)

    kernel: Tensor = polar_wedge_kernel(pad, div, alpha, threshold)
    kernel = kernel * disc_mask
    kernel /= kernel.sum(dim=(1, 2), keepdim=True)

    return kernel
