"""
CenterlineDirectionMetric metric.

This module provides a DirectionMetric class that calculates angular error metrics in evaluation.
"""

from typing import Any, Sequence, Optional, List, Dict

import numpy as np
import torch
import cv2

from mmengine import print_log, MMLogger
from mmengine.evaluator import BaseMetric

from mmseg.registry import METRICS

from prettytable import PrettyTable


@METRICS.register_module()
class CenterlineDirectionMetric(BaseMetric):
    """
    CenterlineDirectionMetric class.

    This class calculates angular error metrics for direction estimation during evaluation.

    Args:
        gt_dir_key (str): Key for ground truth directions in data_samples.
            Default: 'gt_directions'.
        collect_device (str): Device name used for collecting results from different ranks
            during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric names to
            disambiguate homonymous metrics of different evaluators. If prefix is not provided
            in the argument, self.default_prefix will be used instead. Default: ``dir_eval``
        collect_dir: (str, optional): Synchronize directory for collecting data from different
            ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
            Defaults to None.
    """

    default_prefix: str = "dir_eval"

    def __init__(
        self,
        gt_dir_key: str = "gt_directions",
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the CenterlineDirectionMetric class.

        Args:
            gt_dir_key (str): Key for ground truth directions in data_samples.
                Default: 'gt_directions'.
            collect_device (str): Device name used for collecting results from different ranks
                during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
            prefix (str, optional): The prefix that will be added in the metric names to
                disambiguate homonymous metrics of different evaluators. If prefix is not provided
                in the argument, self.default_prefix will be used instead. Default: ``dir_eval``
            collect_dir: (str, optional): Synchronize directory for collecting data from different
                ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
                Defaults to None.
        """
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            collect_dir=collect_dir,
        )
        self.gt_dir_key: str = gt_dir_key

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
            dir_classes: list[int] = data_sample["dir_classes"]
            pred_dir: torch.Tensor = data_sample["estimated_dirs"]["data"]
            gt_centerline: torch.Tensor= data_sample["gt_sem_seg"]["data"]
            class_centerline: torch.Tensor = torch.cat(
                [torch.where(gt_centerline == dir_class, 1, 0) for dir_class in dir_classes]
            )
            gt_dir: torch.Tensor = data_sample[self.gt_dir_key]["data"]

            mask: torch.Tensor = class_centerline > 0

            if not mask.any():
                continue

            gt_vals: torch.Tensor = gt_dir[mask]
            pred_vals: torch.Tensor = pred_dir[mask]

            diff: torch.Tensor = torch.abs(gt_vals - pred_vals)
            errors: torch.Tensor = torch.min(diff, torch.pi - diff)
            errors_deg: torch.Tensor = torch.rad2deg(errors)

            self.results.append(errors_deg.cpu().numpy())

    def compute_metrics(self, results: List[np.ndarray]) -> Dict[str, float | int]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if not results:
            return {}

        errors: np.ndarray = np.concatenate(results)

        n: int = errors.size
        mean_error: float = float(errors.mean())
        median_error: float = float(np.median(errors))
        std_error: float = float(np.std(errors))
        rmse: float = float(np.sqrt(np.mean(errors ** 2)))

        acc_1: float = float((errors <= 1).mean() * 100)
        acc_5: float = float((errors <= 5).mean() * 100)
        acc_10: float = float((errors <= 10).mean() * 100)
        acc_20: float = float((errors <= 20).mean() * 100)

        metrics: dict[str, float | int] = {
            "mean_error": mean_error,
            "median_error": median_error,
            "std_error": std_error,
            "rmse": rmse,
            "acc_1_deg": acc_1,
            "acc_5_deg": acc_5,
            "acc_10_deg": acc_10,
            "acc_20_deg": acc_20,
            "num_pixels": n,
        }

        results_table: PrettyTable = PrettyTable()
        results_table.field_names = ["Metric", "Value"]

        for k, v in metrics.items():
            results_table.add_row([k, f"{v:.2f}" if isinstance(v, float) else v])

        print_log("Direction eval results table:\n" + results_table.get_string(), logger=logger)

        return metrics
