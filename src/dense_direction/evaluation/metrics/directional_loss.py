"""
DirectionalLossMetric metric.

This module provides a DirectionalLossMetric class that evaluates direction estimation
by computing the directional loss on crops of each prediction.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS, MODELS
from mmseg.utils import ConfigType
from torch import Tensor


@METRICS.register_module()
class DirectionalLossMetric(BaseMetric):
    """
    DirectionalLossMetric class.

    Evaluates direction estimation by sliding a fixed-size crop window over each
    prediction and accumulating the directional loss over all foreground pixels.
    The final metric is the pixel-weighted mean loss across all crops and samples.

    The loss function is always configured with ``reduction='none'`` internally so
    that per-pixel losses can be properly accumulated across crops.

    Args:
        loss_config (ConfigType, optional): Loss configuration dict. The loss is forced
            to ``reduction='none'`` regardless of the value in this config.
            Default: ``dict(type="EfficientDirectionalLoss")``.
        dir_classes (Sequence[int] | None, optional): Class indices for which direction
            estimation is performed. Defaults to ``(1,)`` (binary foreground class).
        size (int, optional): Crop size in pixels (height and width). Default: 392.
        step (int, optional): Sliding stride in pixels. Default: 378.
        collect_device (str, optional): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str | None, optional): Metric name prefix. Default: None.
        collect_dir (str | None, optional): Directory for CPU result collection.
            Default: None.
    """

    default_prefix = "directional_loss"
    DEFAULT_LOSS_CFG: dict = dict(type="EfficientDirectionalLoss")

    def __init__(
        self,
        loss_config: ConfigType | None = None,
        dir_classes: Sequence[int] | None = None,
        size: int = 28 * 14,
        step: int = 27 * 14,
        collect_device: str = "cpu",
        prefix: str | None = None,
        collect_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            collect_dir=collect_dir,
        )
        # Force reduction='none' so we can accumulate per-pixel losses ourselves
        cfg = (loss_config or self.DEFAULT_LOSS_CFG).copy()
        cfg["reduction"] = "none"
        self.loss_function = MODELS.build(cfg)
        self._loss_on_device: bool = False

        self.dir_classes: Sequence[int] = dir_classes or (1,)
        self.size: int = size
        self.step: int = step

    # ------------------------------------------------------------------
    # GT transform
    # ------------------------------------------------------------------

    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Converts a multi-class GT map to per-class binary maps.

        Args:
            gt_sem_seg (Tensor): Shape ``(N, C, H, W)``.

        Returns:
            Tensor: Shape ``(N, K, 1, H, W)`` where K = len(dir_classes).
        """
        class_maps = [torch.where(gt_sem_seg == idx, 1, 0) for idx in self.dir_classes]
        return torch.stack(class_maps, dim=1).float()

    # ------------------------------------------------------------------
    # Sliding-window crop iterator
    # ------------------------------------------------------------------

    def _iter_crop_windows(self, h: int, w: int) -> Iterator[tuple[slice, slice]]:
        """
        Yields ``(y_slice, x_slice)`` pairs that tile the image with overlap.

        The last window along each axis is snapped to the image boundary so that
        every pixel is covered without padding.
        """
        h_grids = max(h - self.size + self.step - 1, 0) // self.step + 1
        w_grids = max(w - self.size + self.step - 1, 0) // self.step + 1
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * self.step
                x1 = wi * self.step
                y2 = min(y1 + self.size, h)
                x2 = min(x1 + self.size, w)
                # Snap to boundary so the crop is always exactly self.size
                y1 = max(y2 - self.size, 0)
                x1 = max(x2 - self.size, 0)
                yield slice(y1, y2), slice(x1, x2)

    # ------------------------------------------------------------------
    # Metric protocol
    # ------------------------------------------------------------------

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and accumulate results.

        Each result entry is a ``(loss_sum, pixel_count)`` tuple so that
        ``compute_metrics`` can compute the pixel-weighted mean.
        """
        for data_sample in data_samples:
            pred_vf = data_sample["estimated_vs"]["data"].unsqueeze(0).unsqueeze(0)
            gt_sem_seg = data_sample["gt_sem_seg"]["data"].unsqueeze(0)
            gt_sem_seg = self._transform_gt_sem_seg(gt_sem_seg)

            # Lazy device placement — move loss once on first sample
            if not self._loss_on_device:
                self.loss_function = self.loss_function.to(pred_vf.device)
                self._loss_on_device = True

            h, w = gt_sem_seg.shape[-2:]

            for y_sl, x_sl in self._iter_crop_windows(h, w):
                pred_crop = pred_vf[:, :, :, y_sl, x_sl]
                gt_crop = gt_sem_seg[:, :, :, y_sl, x_sl]

                if gt_crop.sum() == 0:
                    continue

                with torch.no_grad():
                    loss = self.loss_function(pred_crop, gt_crop)

                # loss is (M,) for efficient losses (M = foreground pixels in crop)
                # or (N*K, 1, H, W) for standard losses — handle both
                n_pixels = int(loss.numel())
                if n_pixels == 0:
                    continue

                self.results.append((float(loss.sum().item()), n_pixels))

    def compute_metrics(self, results: list) -> dict:
        """
        Compute the pixel-weighted mean directional loss.

        Args:
            results (list): List of ``(loss_sum, pixel_count)`` tuples.

        Returns:
            dict: ``{"loss": float}`` or ``{"loss": nan}`` when no foreground pixels.
        """
        if not results:
            return {"loss": float("nan")}

        total_loss = sum(r[0] for r in results)
        total_pixels = sum(r[1] for r in results)

        if total_pixels == 0:
            return {"loss": float("nan")}

        return {"loss": total_loss / total_pixels}
