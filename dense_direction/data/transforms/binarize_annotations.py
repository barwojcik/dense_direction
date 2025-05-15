"""
BinarizeAnnotations transform for semantic segmentation masks.

This module provides a binarization transformation that normalizes and threshold the ground truth
semantic segmentation mask by setting values above 0.5 to 1 and others to 0, which is useful for
binary image classification tasks.
"""

import numpy as np
from mmcv import BaseTransform
from mmengine import TRANSFORMS


@TRANSFORMS.register_module()
class BinarizeAnnotations(BaseTransform):
    """
    Performs a binarization transformation on the ground truth semantic segmentation mask.

    Args:
        results (dict): A dictionary containing the ground truth semantic segmentation mask.

    Returns:
        dict: The input dictionary with binarized semantic segmentation mask.
    """

    def transform(self, results: dict) -> dict:
        """
        Applies semantic segmentation mask transformation.

        This method takes in a dictionary of results and applies the following transformations:
        - Normalizes the ground truth semantic segmentation mask by dividing by 3*255
        - Applies thresholding to set values above 0.5 to 1 and others to 0
        - Casts the resulting mask to uint8 data type

        Args:
            results (dict): A dictionary of results containing the ground truth semantic segmentation mask.

        Returns:
            Optional[Union[dict, tuple[list, list]]]: The transformed results or None if no transformation is needed.
        """
        gt_semantic_seg = results["gt_seg_map"]
        gt_semantic_seg = gt_semantic_seg.mean(axis=2, keepdims=True) / 255
        gt_semantic_seg = np.where(gt_semantic_seg > 0.5, 1.0, 0.0)
        results["gt_seg_map"] = gt_semantic_seg
        return results
