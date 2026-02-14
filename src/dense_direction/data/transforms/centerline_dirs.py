"""
CenterlineToDirections transform for direction computation.

This module provides a class for direction map computation from a centerline sematic segmentation
map.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from mmcv import BaseTransform
from mmengine import TRANSFORMS
from scipy import ndimage


@TRANSFORMS.register_module()
class CenterlineToDirections(BaseTransform):
    """
    Computes the direction values for the centerline.

    This transform computes the direction values for the centerline from the ground truth semantic
    segmentation map. Note, the centerline must be stored as a semantic segmentation map under
    `gt_seg_map` in the results dictionary.

    Args:
        sigma_blur (float): Sigma value for the Gaussian blur. Default: 1.0.
        sigma_integration (float): Sigma value for the Gaussian integration. Default: 2.0.
        dir_classes (Sequence[int]): List of class labels for which the directions should be
            computed. If None, it assumes binary data. Default: None.
        result_key (str): Key under which the result will be stored in the results' dictionary.
            Default: `gt_directions`.
    """

    def __init__(
        self,
        sigma_blur: float = 1.0,
        sigma_integration: float = 2.0,
        dir_classes: Sequence[int] | None = None,
        result_key: str = "gt_directions",
    ) -> None:
        """
        Initializes the CenterlineToDirections transform.

        Args:
            sigma_blur (float): Sigma value for the Gaussian blur. Default: 1.0.
            sigma_integration (float): Sigma value for the Gaussian integration. Default: 2.0.
            dir_classes (Sequence[int] | None): List of class labels for which the directions
                should be computed. If None, it assumes binary data. Default: None.
            result_key (str): Key under which the result will be stored in the results' dictionary.
                Default: `gt_directions`.
        """
        super().__init__()
        self.sigma_blur: float = sigma_blur
        self.sigma_integration: float = sigma_integration
        self.result_key: str = result_key
        self.dir_classes: Sequence[int] = dir_classes or (1,)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Computes the direction values for the centerline.

        This method takes in a dictionary of results and computes the direction values for the
        centerline.

        Args:
            results (dict[str, Any]): A dictionary of results containing the ground truth semantic
            segmentation map of the centerline.

        Returns:
            dict[str, Any]: The result dictionary with the computed direction values in radians
                (from 0 to π) for each class in dir_classes. The result is stored under a new key.
        """
        assert (
            "gt_seg_map" in results.keys()
        ), "Missing segmentation map key in results, load annotations first"
        gt_semantic_seg: np.ndarray = results["gt_seg_map"].copy().astype(np.float32)

        assert 2 <= len(gt_semantic_seg.shape) < 4, "Segmentation map should be 2D or 3D"

        class_directions: list[np.ndarray] = []
        for class_idx in self.dir_classes:
            class_mask: np.ndarray = np.where(gt_semantic_seg == class_idx, 1, 0).astype(np.float32)
            blurred_mask: np.ndarray = ndimage.gaussian_filter(class_mask, sigma=self.sigma_blur)

            # Compute gradient magnitude
            Ix: np.ndarray = ndimage.sobel(blurred_mask, axis=1)
            Iy: np.ndarray = -ndimage.sobel(
                blurred_mask, axis=0
            )  # flipped due to an image coordinate system

            # Structure tensor components
            Ixx: np.ndarray = Ix**2
            Ixy: np.ndarray = Ix * Iy
            Iyy: np.ndarray = Iy**2

            # Average the components
            Jxx: np.ndarray = ndimage.gaussian_filter(Ixx, sigma=self.sigma_integration)
            Jxy: np.ndarray = ndimage.gaussian_filter(Ixy, sigma=self.sigma_integration)
            Jyy: np.ndarray = ndimage.gaussian_filter(Iyy, sigma=self.sigma_integration)

            theta: np.ndarray = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
            shifted_theta: np.ndarray = theta + 0.5 * np.pi  # Convert to [0, pi)

            class_directions.append(shifted_theta)

        gt_directions: np.ndarray = np.stack(class_directions, axis=0)
        results[self.result_key] = gt_directions
        return results
