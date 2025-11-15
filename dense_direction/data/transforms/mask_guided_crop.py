"""
Guided Random Crop Transformation for images and semantic segmentation maps.

This module provides a class for transformation that randomly crops images and semantic
segmentation maps, while ensuring that the positive class is in the middle of the image and does
not take too much or too little image area.
"""

import random
import numpy as np
from typing import Union, Optional
from mmcv import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine import TRANSFORMS


@TRANSFORMS.register_module()
class MaskGuidedRandomCrop(BaseTransform):
    """
    Random crop the image and sematic segmentation map based on positive class location.

    This transform ensures that the positive class is in the middle of the image and does not
    take too much or too little image area.

    Required Keys:
        - img
        - gt_seg_map

    Modified Keys:
        - img
        - img_shape
        - gt_seg_map

    Args:
        crop_size (Union[int, Tuple[int, int]]): Expected size after cropping with the format of
            (h, w). If set to an integer, then cropping width and height are equal to this integer.
        min_ratio (float): The minimum ratio that a positive category can occupy.
        max_ratio (float): The maximum ratio that a positive category can occupy.
        max_attempts (int): The maximum number of attempts to crop.
        ignore_index (int): The label index to be ignored. Default: 255
        by_index (Optional[int]): The index of the category to be used to guide cropping. If None,
            it assumes binary data. Default: None.
    """

    def __init__(
        self,
        crop_size: Union[int, tuple[int, int]],
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
        max_attempts: int = 100,
        ignore_index: int = 255,
        by_index: Optional[int] = None,
    ) -> None:
        """
        Initializes the object with parameters for guided cropping.

        Args:
            crop_size (Union[int, Tuple[int, int]]): Expected size after cropping with the format
                of (h, w). If set to an integer, then cropping width and height are equal to this
                integer.
            min_ratio (float): The minimum ratio that a positive category can occupy.
            max_ratio (float): The maximum ratio that a positive category can occupy.
            max_attempts (int): The maximum number of attempts to crop.
            ignore_index (int): The label index to be ignored. Default: 255
            by_index (Optional[int]): The index of the category to be used to guide cropping.
                If None, it assumes binary data. Default: None.

        Raises:
            AssertionError: If the provided crop_size is invalid or if any ratio values are
                outside their expected ranges.
        """
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), "The expected crop_size is an integer, or a tuple containing two integers"

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        assert crop_size[0] > 0 and crop_size[1] > 0
        assert 0 <= min_ratio <= 1
        assert 0 <= max_ratio <= 1

        self.crop_size: tuple[int, int] = crop_size
        self.margins: tuple[int, int, int, int] = (
            int(crop_size[0] / 2),
            int(crop_size[1] / 2),
            crop_size[0] - int(crop_size[0] / 2),
            crop_size[1] - int(crop_size[1] / 2),
        )
        self.min_ratio: float = min_ratio
        self.max_ratio: float = max_ratio
        self.max_attempts: int = max_attempts
        self.ignore_index: int = ignore_index
        self.by_index: Optional[int] = by_index

    def _evaluate_crop_location(self, crop_location: tuple[int, int], mask: np.ndarray) -> bool:
        """
        Evaluates whether the given crop location is within an acceptable ratio of positive pixels.

        Args:
            crop_location (tuple[int, int]): The coordinates of the crop location.
            mask (np.ndarray): The image mask.

        Returns:
            bool: True if the crop location is within the acceptable ratio, False otherwise.
        """

        mask_crop: np.ndarray = self._crop_image(mask, crop_location)
        positive_ratio: float = mask_crop.mean()
        if self.min_ratio <= positive_ratio <= self.max_ratio:
            return True
        return False

    @cache_randomness
    def _get_crop_location(self, results: dict) -> tuple[int, int]:
        """
        Get a location (y, x) of the upper-left corner of the cropped image.

        Args:
            results (dict): Result dict from a loading pipeline.

        Returns:
            tuple[int, int]: Coordinates of the upper left corner (y, x).
        """

        my1, my2, mx1, mx2 = self.margins
        mask_roi: np.ndarray = results["gt_seg_map"][my1 : -(my2 + 1), mx1 : -(mx2 + 1)]
        if self.by_index is not None:
            mask_roi = np.where(mask_roi == self.by_index, 1, 0)

        location_candidates: list[tuple[int, int]] = list(zip(*np.where(mask_roi > 0)))
        assert location_candidates, "No positive class found in the mask"

        crop_location: tuple[int, int] = random.choice(location_candidates)

        if self.min_ratio == 0.0 and self.max_ratio == 1.0:
            return crop_location

        if self._evaluate_crop_location(crop_location, results["gt_seg_map"]):
            return crop_location

        for _ in range(self.max_attempts):
            crop_location = random.choice(location_candidates)
            if self._evaluate_crop_location(crop_location, results["gt_seg_map"]):
                return crop_location

        return crop_location

    def _crop_image(self, image: np.ndarray, crop_location: tuple[int, int]) -> np.ndarray:
        """
        Crop from image based on the provided bounding box.

        Args:
            image (np.ndarray): Original input image.
            crop_location (tuple): Coordinates of the upper left corner (y, x).

        Returns:
            np.ndarray: The cropped image.
        """

        y1, x1 = crop_location
        y2, x2 = y1 + self.crop_size[0], x1 + self.crop_size[1]
        image = image[y1:y2, x1:x2, ...]
        return image

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from a loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        crop_location: tuple[int, int] = self._get_crop_location(results)

        # Crop image and alter the image and its shape in results
        img: np.ndarray = self._crop_image(results["img"], crop_location)
        results["img"] = img
        results["img_shape"] = img.shape[:2]

        # Crop all segmentation maps
        for key in results.get("seg_fields", ["gt_seg_map"]):
            results[key] = self._crop_image(results[key], crop_location)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"
