"""
Local Visualizer for semantic segmentation with direction estimation.

This module contains the SegDirLocalVisualizer class, which is used to visualize the results of a
segmentation and direction estimation. It's a counterpart to `mmseg` SegLocalVisualizer class,
which is used to visualize the results of a semantic segmentation or depth estimation.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

import mmcv
from mmengine.dist import master_only
from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@VISUALIZERS.register_module()
class SegDirLocalVisualizer(SegLocalVisualizer):
    """
    Local Visualizer for semantic segmentation with direction estimation.

    This class is used to visualize the results of a segmentation and direction estimation.
    It inherits from `mmseg`'s SegLocalVisualizer and adds the function of drawing a map of
    estimated directions. It works for either separate sematic segmentation/direction estimation
    or both tasks done joinly.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format should be RGB.
            Defaults to None.
        vis_backends (list, optional): Visual backend config list. Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends. If it is None, the
            backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the prediction of a
            segmentation model is a segment map with label indices, `classes` is a list that
            includes items responding to the label indices. If classes are not defined, visualizer
            will take cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is a list of color
            palettes responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias` visualizer will use the
            meta-information of the dataset i.e., classes and palette, but the `classes` and
            `palette` have higher priority. Defaults to None.
        alpha (int, float): The transparency of the segmentation mask. Defaults to 0.8.
    """

    def __init__(
        self,
        name: str = "visualizer",
        image: Optional[np.ndarray] = None,
        vis_backends: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        classes: Optional[List] = None,
        palette: Optional[List] = None,
        dataset_name: Optional[str] = None,
        alpha: float = 0.8,
        **kwargs
    ) -> None:
        """
        Initializes SegDirLocalVisualizer class.

        This class is used to visualize the results of a segmentation and direction estimation.
        It inherits from `mmseg`'s SegLocalVisualizer and adds the function of drawing a map of
        estimated directions. It works for either separate sematic segmentation/direction estimation
        or both tasks done joinly.

        Args:
            name (str): Name of the instance. Defaults to 'visualizer'.
            image (np.ndarray, optional): the origin image to draw. The format should be RGB.
                Defaults to None.
            vis_backends (list, optional): Visual backend config list. Defaults to None.
            save_dir (str, optional): Save file dir for all storage backends. If it is None, the
                backend storage will not save any data.
            classes (list, optional): Input classes for result rendering, as the prediction of a
                segmentation model is a segment map with label indices, `classes` is a list that
                includes items responding to the label indices. If classes are not defined, visualizer
                will take cityscapes` classes by default. Defaults to None.
            palette (list, optional): Input palette for result rendering, which is a list of color
                palettes responding to the classes. Defaults to None.
            dataset_name (str, optional): `Dataset name or alias` visualizer will use the
                meta-information of the dataset i.e., classes and palette, but the `classes` and
                `palette` have higher priority. Defaults to None.
            alpha (int, float): The transparency of the segmentation mask. Defaults to 0.8.
        """
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            alpha=alpha,
            **kwargs,
        )
        self.set_dataset_meta(
            palette=palette,
            classes=classes,
            dataset_name=dataset_name,
        )

    def _draw_dir_map(
        self,
        image: np.ndarray,
        dir_map: np.ndarray,
    ) -> np.ndarray:
        """
        Draws a map of directions on a given image.

        Args:
            image (np.ndarray): Image to draw directions on.
            dir_map (np.ndarray): Estimated directions.

        Returns:
            np.ndarray: Image with overlay of estimated directions on it.
        """

        dir_map = (dir_map / np.pi * 255).astype(np.uint8)
        dir_map = cv2.applyColorMap(dir_map, cv2.COLORMAP_HSV)
        dir_map = cv2.cvtColor(dir_map, cv2.COLOR_BGR2RGB)
        dir_map = dir_map * self.alpha + image * (1 - self.alpha)
        return dir_map.astype(np.uint8)

    def _draw_dir_maps(
        self,
        image: np.ndarray,
        data_sample: SegDataSample,
    ) -> np.ndarray:
        """
        Draws direction maps on a given image.

        Args:
            image (np.ndarray): Image to draw directions on.
            data_sample (SegDataSample): PixelData of estimated directions.

        Returns:
            np.ndarray: Image with overlay of estimated directions on it.
        """

        estimated_dirs: np.ndarray = data_sample.estimated_dirs.cpu().data.numpy()
        dir_maps: list[np.ndarray] = [
            self._draw_dir_map(image, dir_map) for dir_map in estimated_dirs
        ]

        if "gt_sem_seg" in data_sample and "dir_classes" in data_sample:
            gt_sem_seg: np.ndarray = data_sample.gt_sem_seg.cpu().data.numpy()
            masks: list[np.ndarray] = [
                np.where(gt_sem_seg == class_idx, 1, 0) for class_idx in data_sample.dir_classes
            ]
            masks = [np.concatenate(3 * [mask.transpose(1, 2, 0)], axis=-1) for mask in masks]
            dir_maps = [
                np.where(mask == 0, image, dir_map) for mask, dir_map in zip(masks, dir_maps)
            ]

        out_map: np.ndarray = np.concatenate(dir_maps, axis=1)
        self.set_image(out_map)

        return out_map

    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional[SegDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        step: int = 0,
        with_labels: Optional[bool] = True,
    ) -> None:
        """
        Draw a data sample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed in a stitched image
        where the left image is the ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images will be displayed
        in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to ``out_file``. It is
        usually used when the display is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`SegDataSample`, optional): SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample. Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to the output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            with_labels(bool, optional): Add semantic labels in a visualization result.
                Defaults to True.
        """

        if not data_sample:
            self.set_image(image)
            self.add_image(name, image, step)
            return

        classes = self.dataset_meta.get("classes", None)
        palette = self.dataset_meta.get("palette", None)

        imgs_to_draw: list[np.ndarray] = []

        if draw_gt:
            if "gt_sem_seg" in data_sample:
                assert classes is not None, (
                    "class information is not provided when visualizing semantic segmentation "
                    "results."
                )
                imgs_to_draw.append(
                    self._draw_sem_seg(image, data_sample.gt_sem_seg, classes, palette, with_labels)
                )

        if draw_pred:
            if "pred_sem_seg" in data_sample:
                assert classes is not None, (
                    "class information is not provided when visualizing semantic segmentation "
                    "results."
                )
                imgs_to_draw.append(
                    self._draw_sem_seg(
                        image, data_sample.pred_sem_seg, classes, palette, with_labels
                    )
                )

            if "estimated_dirs" in data_sample:
                imgs_to_draw.append(self._draw_dir_maps(image, data_sample))

        imgs_to_draw = [img for img in imgs_to_draw if img is not None]
        drawn_img: np.ndarray = np.concatenate(imgs_to_draw, axis=1)
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
