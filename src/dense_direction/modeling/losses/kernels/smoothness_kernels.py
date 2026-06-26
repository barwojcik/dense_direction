"""Smoothness loss kernel functions.

This module provides spatial weight kernels for use with SmoothnessLoss.
Each kernel function returns a 2D weight array of shape (k_size, k_size) that
describes how to weight neighbors when aggregating direction vectors in a local
neighborhood.

All kernels are registered with FUNCTIONS so they can be referenced by name in
config dicts (e.g. ``dict(type="gaussian_smoothness_kernel", sigma=1.0)``).
"""

from typing import Any

import numpy as np
import torch
from mmengine import FUNCTIONS
from torch import Tensor

from .utils import get_coordinates, get_disc_mask, get_kernel_size


@FUNCTIONS.register_module()
def gaussian_smoothness_kernel(
    pad: int = 1,
    sigma: float | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Returns a 2D Gaussian spatial weight kernel.

    Produces a kernel where the center pixel has weight 1 and weights decay
    smoothly towards the edges following a Gaussian profile. This gives a soft,
    distance-sensitive neighborhood for the smoothness loss.

    Args:
        pad (int): Half-size of the kernel; kernel size = 2 * pad + 1. Default: 1.
        sigma (float | None): Standard deviation of the Gaussian. Defaults to
            ``pad / 2`` if None, which places roughly 95% of weight within
            distance ``pad``.
        **kwargs: Ignored extra keyword arguments (for config compatibility).

    Returns:
        Tensor: Gaussian weight kernel of shape (k_size, k_size).
    """
    k_size = get_kernel_size(pad)
    if sigma is None:
        sigma = pad / 2.0 if pad > 0 else 1.0

    coords = get_coordinates(k_size)  # (2, k_size, k_size)
    dist_sq = (coords**2).sum(0)  # (k_size, k_size)
    kernel = np.exp(-dist_sq / (2.0 * sigma**2))
    # Zero out the center so a pixel does not vote for itself
    center = pad
    kernel[center, center] = 0.0
    return torch.tensor(kernel, dtype=torch.float32)


@FUNCTIONS.register_module()
def uniform_smoothness_kernel(
    pad: int = 1,
    **kwargs: Any,
) -> Tensor:
    """
    Returns a uniform disc-shaped spatial weight kernel.

    All neighbors within the disc of radius ``pad`` receive equal weight 1;
    the center pixel receives weight 0 (a pixel should not count as its own
    neighbor). This matches the default implicit behavior of SmoothnessLoss.

    Args:
        pad (int): Half-size of the kernel; kernel size = 2 * pad + 1. Default: 1.
        **kwargs: Ignored extra keyword arguments (for config compatibility).

    Returns:
        Tensor: Uniform disc kernel of shape (k_size, k_size).
    """
    k_size = get_kernel_size(pad)
    kernel = get_disc_mask(k_size).astype(np.float32)
    # Zero out the center so a pixel does not vote for itself
    center = pad
    kernel[center, center] = 0.0
    return torch.tensor(kernel, dtype=torch.float32)


@FUNCTIONS.register_module()
def distance_weighted_smoothness_kernel(
    pad: int = 1,
    power: float = 1.0,
    **kwargs: Any,
) -> Tensor:
    """
    Returns an inverse-distance spatial weight kernel.

    Closer neighbors receive higher weights according to ``1 / dist^power``.
    The center pixel is excluded (weight 0). A ``power`` of 1 gives the classic
    inverse-distance weighting; larger values further emphasize nearby pixels.

    Args:
        pad (int): Half-size of the kernel; kernel size = 2 * pad + 1. Default: 1.
        power (float): Distance exponent. Must be > 0. Default: 1.0.
        **kwargs: Ignored extra keyword arguments (for config compatibility).

    Returns:
        Tensor: Inverse-distance weight kernel of shape (k_size, k_size).
    """
    assert power > 0, f"power must be positive, got {power}"
    k_size = get_kernel_size(pad)
    coords = get_coordinates(k_size)  # (2, k_size, k_size)
    dist = np.sqrt((coords**2).sum(0))  # (k_size, k_size)
    # Avoid division by zero at center; weight will be set to 0 anyway
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(dist > 0, 1.0 / (dist**power), 0.0)
    return torch.tensor(kernel.astype(np.float32), dtype=torch.float32)
