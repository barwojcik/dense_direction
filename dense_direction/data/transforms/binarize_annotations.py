"""
BinarizeAnnotations transform for semantic segmentation maps.

This module provides a class for binarization transformation that normalizes and thresholds the
ground truth semantic segmentation map by setting values above 0.5 to 1 and others to 0, which
is required for direction estimation.
"""

import numpy as np
from typing import Any
from mmcv import BaseTransform
from mmengine import TRANSFORMS


@TRANSFORMS.register_module()
class BinarizeAnnotations(BaseTransform):
    """
    Performs a binarization transformation on the ground truth semantic segmentation map.

    This class applies the following transformations to the semantic segmentation map:
        - Reduces the channel number to 1 and dimensions to 2
        - Normalizes values to 0-1 range
        - Applies thresholding to set values above 0.5 to 1 and others to 0
    """

    MAX_PIXEL_VALUE = 255
    THRESHOLD = 0.5

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Applies semantic segmentation map transformation.

        This method takes in a dictionary of results and applies the following transformations
        to the semantic segmentation map:
            - Reduces the channel number to 1 and dimensions to 2
            - Normalizes values to 0-1 range
            - Applies thresholding to set values above 0.5 to 1 and others to 0

        Args:
            results (dict[str, Any]): A dictionary of results containing the ground truth semantic
            segmentation map.

        Returns:
            dict[str, Any]: The result dictionary with the transformed semantic
            segmentation map.
        """
        assert (
            "gt_seg_map" in results.keys()
        ), "Missing segmentation map key in results, load annotations first"
        gt_semantic_seg = results["gt_seg_map"]

        assert (
            len(gt_semantic_seg.shape) >= 2 or len(gt_semantic_seg.shape) < 4
        ), "Segmentation map should be 2D or 3D"

        # normalize values
        if len(gt_semantic_seg.shape) == 2:
            gt_semantic_seg = gt_semantic_seg / self.MAX_PIXEL_VALUE
        else:
            gt_semantic_seg = gt_semantic_seg.mean(axis=2) / self.MAX_PIXEL_VALUE

        # binarize values
        gt_semantic_seg = np.where(gt_semantic_seg > self.THRESHOLD, 1.0, 0.0)

        results["gt_seg_map"] = gt_semantic_seg
        return results
