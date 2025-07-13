"""
DirectionalLossMetric metric.

This module provides a DirectionalLossMetric class that calculates loss value in evaluation.
"""

from typing import Any, Callable, Sequence, Optional

import numpy as np

import torch
from torch import Tensor

from mmengine import FUNCTIONS
from mmengine.evaluator import BaseMetric
from mmseg.evaluation.metrics import CityscapesMetric, IoUMetric, DepthMetric
from mmseg.registry import MODELS, METRICS
from mmseg.utils import ConfigType


@METRICS.register_module()
class DirectionalLossMetric(BaseMetric):
    """
    DirectionalLossMetric class.

    This class that calculates directional loss during evaluation.

    Args:
        loss_config (ConfigType): Loss configuration.
        dir_classes (int|Sequence[int]): Index or sequence of indexes of classes for which
            direction estimation is performed. If not provided, it assumes binary segmentation and
            positive class as linear. Default: None.
        collect_device (str): Device name used for collecting results from different ranks
            during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric names to
            disambiguate homonymous metrics of different evaluators. If prefix is not provided
            in the argument, self.default_prefix will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data from different
            ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
            Defaults to None.
    """

    default_prefix = "directional_loss"
    DEFAULT_LOSS_CFG = dict(type="EfficientDirectionalLoss")

    def __init__(
        self,
        loss_config: Optional[ConfigType] = None,
        dir_classes: Sequence[int] = None,
        size: int = 28*14,
        step: int = 27*14,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes DirectionalLossMetric class.

        This class that calculates directional loss during evaluation.

        Args:
            loss_config (ConfigType): Loss configuration.
            dir_classes (int|Sequence[int]): Index or sequence of indexes of classes for which
                direction estimation is performed. If not provided, it assumes binary segmentation and
                positive class as linear. Default: None.
            collect_device (str): Device name used for collecting results from different ranks
                during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
            prefix (str, optional): The prefix that will be added in the metric names to
                disambiguate homonymous metrics of different evaluators. If prefix is not provided
                in the argument, self.default_prefix will be used instead. Default: None
            collect_dir: (str, optional): Synchronize directory for collecting data from different
                ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
                Defaults to None.
        """
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            collect_dir=collect_dir,
        )
        self.loss_function = MODELS.build(loss_config or self.DEFAULT_LOSS_CFG).cuda()
        self.dir_classes: Sequence[int] = dir_classes or (1,)
        self.size: int = size
        self.step: int = step

    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Transforms gt_sem_seg maps into separate binary maps per class for which directions will
        be estimated.

        Args:
            gt_sem_seg (Tensor): Ground truth semantic segmentation map of shape (N, C, H, W).

        Returns:
            Tensor: Semantic segmentation map of shape (N, K, 1, H, W) where K number of linear
                classes.
        """
        class_maps: list[Tensor] = []
        for class_index in self.dir_classes:
            class_map: Tensor = torch.where(gt_sem_seg == class_index, 1, 0)
            class_maps.append(class_map)
        return torch.stack(class_maps, dim=1).float()

    def compute_metrics(self, results: list) -> dict:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed loss.
        """
        loss_value = sum([r[0] for r in results])/sum([r[1] for r in results])
        return {'loss': loss_value}

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_vector_field = data_sample["estimated_vs"]["data"].unsqueeze(0).unsqueeze(0)
            gt_sem_seg = data_sample["gt_sem_seg"]["data"].unsqueeze(0)
            gt_sem_seg = self._transform_gt_sem_seg(gt_sem_seg)

            # TODO: fix this, tbh the whole class needs a rewrite
            x = 0
            y = 0
            h, w = gt_sem_seg.shape[-2:]

            while x < h:
                while y < w:
                    pred_crop = pred_vector_field[:, :, :, x:x+self.size, y:y+self.size]
                    gt_crop = gt_sem_seg[:, :, :, x:x+self.size, y:y+self.size]

                    if gt_crop.sum() > 0:
                        loss = self.loss_function(pred_crop, gt_crop)
                        self.results.append((loss.sum().item(), len(loss)))

                    y = y + self.step

                x = x + self.step
