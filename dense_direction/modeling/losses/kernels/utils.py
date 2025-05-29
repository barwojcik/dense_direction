"""
Utility functions for kernel computation.
"""

import numpy as np


def get_coordinates(k_size: int) -> np.ndarray:
    """
    Returns 2D array filled with coordinates.

    Returns 2D array filled with x, y coordinates, assuming (0, 0) is in the center of the array.

    Returns:
        np.ndarray: 2D array filled with coordinates.
    """
    cords = np.fromfunction(lambda x, y: np.asarray([x, y]), (k_size, k_size))
    cords = cords - (k_size - 1) / 2
    return cords


def get_polar_coordinates(k_size: int) -> np.ndarray:
    """
    Returns 2D array filled with polar coordinates.

    Returns 2D array filled with angles, assuming (0, 0) is in the center of the array.

    Returns:
        np.ndarray: 2D array filled with polar coordinates.
    """
    center = (k_size - 1) / 2
    cords = np.fromfunction(lambda x, y: np.arctan2(x - center, y - center), (k_size, k_size))
    return cords


def get_disc_mask(k_size: int, radius: int = None) -> np.ndarray:
    """
    Returns 2D disc mask of given radius.

    Returns a disc mask

    Returns:
        np.ndarray: 2D disc mask of given radius.
    """
    radius = radius or (k_size - 1) / 2
    assert radius <= (k_size - 1) / 2

    cords = get_coordinates(k_size)
    distances = np.sqrt((cords**2).sum(0))
    mask = np.where(distances <= radius, 1, 0)
    return mask
