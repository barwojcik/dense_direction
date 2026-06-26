"""
Efficient Von Mises Directional Loss class.

Memory-efficient variant of VonMisesDirectionalLoss that uses F.unfold + F.linear
and only processes foreground pixels, avoiding the full (N*K, div, H, W) intermediate.
"""

from typing import Any

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import Tensor

from .base_circular_directional import EfficientBaseCircularDirectionalLoss


@MODELS.register_module()
class EfficientVonMisesDirectionalLoss(EfficientBaseCircularDirectionalLoss):
    """
    EfficientVonMisesDirectionalLoss class.

    Memory-efficient variant of ``VonMisesDirectionalLoss``.  Uses ``F.unfold`` +
    ``F.linear`` and only computes the von Mises NLL for foreground pixels, which
    reduces peak memory usage on large images.

    See ``VonMisesDirectionalLoss`` for the mathematical definition.

    Args:
        pad (int, optional): Pad size for kernels. Default: 3.
        div (int, optional): Number of direction bins. Default: 20.
        mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
        kappa (float, optional): Von Mises concentration. Higher = sharper. Default: 8.0.
        doubled_angles (bool, optional): Double angles for 180° periodicity. Default: True.
        kernel_cfg (ConfigType, optional): Kernel config.
            Default: ``dict(type="circular_point_kernel")``.
        reduction (str, optional): ``'mean'``, ``'sum'``, or ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Global loss weight. Default: 1.0.
        loss_name (str, optional): Name for logging. Default: ``"loss_vm_dir"``.
    """

    def __init__(
        self,
        kappa: float = 8.0,
        doubled_angles: bool = True,
        loss_name: str = "loss_vm_dir",
        **kwargs: Any,
    ) -> None:
        super().__init__(loss_name=loss_name, **kwargs)
        self.kappa: float = kappa
        self.doubled_angles: bool = doubled_angles

        angle_scale = 2 * np.pi if doubled_angles else np.pi
        bins = torch.arange(0, self.div) / self.div * angle_scale
        self.register_buffer("direction_bins", bins.reshape(1, self.div).float())

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes efficient Von Mises directional loss.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.
            gt_sem_seg (Tensor): Shape ``(N, K, 1, H, W)``.
            weight (float | None): Optional external loss weight.

        Returns:
            Tensor: Von Mises directional loss value.
        """
        pred_flat = self._convert_to_direction_flat(pred_vector_field)  # (N*K*H*W,)
        gt_sq = gt_sem_seg.squeeze(2)  # (N, K, H, W)
        mask_idx = self._get_mask_idx(gt_sq)  # (M,)

        if mask_idx.numel() == 0:
            return pred_vector_field.sum() * 0.0

        direction_values = self._transform_gt_sem_seg_efficient(gt_sq, mask_idx)  # (M, div)
        filtered_pred = pred_flat.index_select(0, mask_idx)  # (M,)

        # Weighted circular mean of GT angles
        weights = direction_values / (direction_values.sum(-1, keepdim=True) + self.EPS)
        gt_sin = (weights * torch.sin(self.direction_bins)).sum(-1)  # (M,)
        gt_cos = (weights * torch.cos(self.direction_bins)).sum(-1)
        gt_angle = torch.atan2(gt_sin, gt_cos)

        pred_angle_d = filtered_pred * 2.0 if self.doubled_angles else filtered_pred
        delta = pred_angle_d - gt_angle
        loss = (-self.kappa * torch.cos(delta) + self.kappa) / (2 * self.kappa)

        return self._reduce_flat(loss, weight)
