"""
CenterlineDirectionMetric metric.

This module provides a DirectionMetric class that calculates angular error metrics in evaluation.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from mmengine import MMLogger, print_log
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
        acc_thresholds (Sequence[float]): Thresholds for calculating accuracy metrics.
            Default: (1, 5, 10, 20).
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
    _DEFAULT_ACC_THRSH: tuple[float, ...] = (1, 5, 10, 20)

    def __init__(
        self,
        gt_dir_key: str = "gt_directions",
        acc_thresholds: Sequence[float] | None = None,
        collect_device: str = "cpu",
        prefix: str | None = None,
        collect_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the CenterlineDirectionMetric class.

        Args:
            gt_dir_key (str): Key for ground truth directions in data_samples.
                Default: 'gt_directions'.
            acc_thresholds (Sequence[float]): Thresholds for calculating accuracy metrics.
                Default: (1, 5, 10, 20).
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
        self.acc_thresholds: Sequence[float] = acc_thresholds or self._DEFAULT_ACC_THRSH

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
            gt_centerline: torch.Tensor = data_sample["gt_sem_seg"]["data"]
            class_centerline: torch.Tensor = torch.cat(
                [(gt_centerline == dir_class) for dir_class in dir_classes]
            )
            gt_dir: torch.Tensor = data_sample[self.gt_dir_key]["data"]

            mask: torch.Tensor = class_centerline > 0

            if not mask.any():
                continue

            gt_vals: torch.Tensor = gt_dir[mask]
            pred_vals: torch.Tensor = pred_dir[mask]

            diff: torch.Tensor = torch.abs(gt_vals - pred_vals)
            errors: torch.Tensor = torch.minimum(diff, torch.pi - diff)
            errors_deg: torch.Tensor = torch.rad2deg(errors)

            channel_labels: torch.Tensor = torch.as_tensor(dir_classes).to(pred_dir.device)
            label_map: torch.Tensor = torch.ones_like(pred_dir) * channel_labels.reshape(-1, 1, 1)
            label_map = label_map.int()

            errors_labels: torch.Tensor = label_map[mask]
            errors_dict: dict[int, np.ndarray] = {
                dir_class: errors_deg[errors_labels == dir_class].cpu().numpy()
                for dir_class in dir_classes
            }

            self.results.append(errors_dict)

    @staticmethod
    def _aggregate_results(results: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
        """Aggregate results from multiple batches."""
        aggregated_results: dict[int, list] = defaultdict(list)
        for result in results:
            for dir_class, errors in result.items():
                aggregated_results[dir_class].append(errors)
        return {
            dir_class: np.concatenate(errors) for dir_class, errors in aggregated_results.items()
        }

    def _compute_metrics(self, errors: np.ndarray) -> dict[str, float | int]:
        """Compute metrics from errors."""
        n: int = errors.size
        mean_error: float = float(errors.mean())
        median_error: float = float(np.median(errors))
        std_error: float = float(np.std(errors))
        rmse: float = float(np.sqrt(np.mean(errors**2)))

        accs = {
            f"acc_{acc_thr:.1f}_deg": float((errors <= acc_thr).mean() * 100)
            for acc_thr in self.acc_thresholds
        }

        return {
            "mean_error": mean_error,
            "median_error": median_error,
            "std_error": std_error,
            "rmse": rmse,
            **accs,
            "num_pixels": n,
        }

    def compute_metrics(self, results: list[dict[int, np.ndarray]]) -> dict[str, float | int]:
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

        aggregated_results: dict[int, np.ndarray] = self._aggregate_results(results)
        errors: np.ndarray = np.concatenate([result for result in aggregated_results.values()])

        overall_metrics: dict[str, float] = self._compute_metrics(errors)

        results_table: PrettyTable = PrettyTable()
        results_table.field_names = [
            "Classes",
        ] + [k for k in overall_metrics.keys()]
        results_table.add_row(
            [
                "Overall",
                *[f"{v:.2f}" if isinstance(v, float) else v for v in overall_metrics.values()],
            ]
        )

        if len(aggregated_results) > 1:
            for dir_class, errors in aggregated_results.items():
                class_metrics: dict[str, float] = self._compute_metrics(errors)

                results_table.add_row(
                    [
                        f"Class {dir_class}",
                        *[
                            f"{v:.2f}" if isinstance(v, float) else v
                            for v in class_metrics.values()
                        ],
                    ]
                )

        print_log(
            "Centerline direction eval results table:\n" + results_table.get_string(), logger=logger
        )

        return overall_metrics
