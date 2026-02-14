"""
Utility functions for kernel computation.
"""

import numpy as np


def get_kernel_size(pad: int) -> int:
    """
    Returns the kernel size for a given pad.

    Args:
        pad (int): The pad.

    Returns:
        int: The kernel size.
    """

    return 2 * pad + 1


def get_coordinates(size: int) -> np.ndarray:
    """
    Returns 2D array filled with coordinates.

    Returns 2D array filled with x, y coordinates, assuming (0, 0) is in the center of the array.

    Args:
        size (int): The dimension of height and width of the 2D array.

    Returns:
        np.ndarray: 3D array filled with coordinates.
    """

    assert size > 0, f"Size should be a positive integer, instead got {size}"

    cords = np.fromfunction(lambda x, y: np.asarray([x, y]), (size, size))
    cords = cords - (size - 1) / 2
    return cords


def get_stacked_coordinates(size: int, division: int) -> np.ndarray:
    """
    Returns stack of 3D arrays filled with coordinates.

    Args:
        size (int): The dimension of height and width of the 2D array.
        division (int): The number of divisions along the 2D array.

    Returns:
        np.ndarray: 4D array filled with coordinates.
    """

    coordinates: np.ndarray = get_coordinates(size)
    return np.stack(division * [coordinates], axis=1)


def get_polar_coordinates(size: int) -> np.ndarray:
    """
    Returns 2D array filled with polar coordinates.

    Returns 2D array filled with angles, assuming (0, 0) is in the center of the array.

    Args:
        size (int): The dimension of height and width of the 2D array.

    Returns:
        np.ndarray: 2D array filled with polar coordinates.
    """

    assert size > 0, f"Size should be a positive integer, instead got {size}"

    center = (size - 1) / 2
    cords = np.fromfunction(lambda x, y: np.arctan2(x - center, y - center), (size, size))
    return cords


def get_disc_mask(size: int, radius: float | None = None) -> np.ndarray:
    """
    Returns 2D disc mask of a given radius.

    Returns a mask in the shape of a disc.

    Args:
        size (int): The dimension of height and width of the 2D array.
        radius (float | None): The radius of the disc mask. Defaults to half of the size.

    Returns:
        np.ndarray: 2D disc mask of a given radius.
    """

    assert size > 0, f"Size should be a positive integer, instead got {size}"

    max_radius = (size - 1) / 2
    radius = radius or max_radius
    assert radius <= max_radius, f"Radius has to be smaller than {max_radius}, got {radius}"

    cords = get_coordinates(size)
    distances = np.sqrt((cords**2).sum(0))
    mask = np.where(distances <= radius, 1, 0)
    return mask


def get_points_on_semicircle(division: int) -> np.ndarray:
    """
    Returns 2D points placed evenly along a semicircle.

    Args:
        division (int): The number of points along a semicircle.

    Returns:
        np.ndarray: 2D array filled with point coordinates.
    """

    assert division > 0, f"Division should be a positive integer, instead got {division}"

    angles = np.linspace(0, np.pi, division, False)
    return np.stack([np.sin(angles), np.cos(angles)])
