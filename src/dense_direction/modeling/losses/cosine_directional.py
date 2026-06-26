"""
Cosine Directional Loss class.

This module contains CosineDirectionalLoss, a directional loss based on cosine regression.
It computes the weighted circular mean of the GT direction distribution derived from kernel
convolutions and measures the cosine distance to the predicted direction. Doubled angles
handle the 180° periodicity of undirected orientations.
"""

from typing import Any

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import Tensor

from .base_circular_directional import BaseCircularDirectionalLoss


@MODELS.register_module()
class CosineDirectionalLoss(BaseCircularDirectionalLoss):
    """
    CosineDirectionalLoss class.

    Computes a cosine regression loss between the predicted direction and the weighted
    circular mean of the GT direction distribution derived from kernel convolutions.

    Loss = 1 - dot(pred_unit, gt_unit)

    where both unit vectors are in the doubled-angle space so that the 180° periodicity
    of undirected orientations is handled correctly.

    Args:
        pad (int, optional): Pad size for kernels. Default: 3.
        div (int, optional): Number of direction bins. Default: 20.
        mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
        doubled_angles (bool, optional): Double angles for 180° periodicity. Default: True.
        kernel_cfg (ConfigType, optional): Kernel config.
            Default: ``dict(type="circular_point_kernel")``.
        reduction (str, optional): ``'mean'``, ``'sum'``, or ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Global loss weight. Default: 1.0.
        loss_name (str, optional): Name for logging. Default: ``"loss_cos_dir"``.
    """

    def __init__(
        self,
        doubled_angles: bool = True,
        loss_name: str = "loss_cos_dir",
        **kwargs: Any,
    ) -> None:
        super().__init__(loss_name=loss_name, **kwargs)
        self.doubled_angles: bool = doubled_angles

        angle_scale = 2 * np.pi if doubled_angles else np.pi
        bins = torch.arange(0, self.div) / self.div * angle_scale
        self.register_buffer("direction_bins", bins.reshape(1, self.div, 1, 1).float())

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes cosine directional loss.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.
            gt_sem_seg (Tensor): Shape ``(N, K, 1, H, W)``.
            weight (float | None): Optional external loss weight.

        Returns:
            Tensor: Cosine directional loss value.
        """
        h, w = pred_vector_field.shape[-2:]
        pred = pred_vector_field.reshape(-1, 2, h, w)
        gt = gt_sem_seg.reshape(-1, 1, h, w)

        pred_angle = self._convert_to_direction(pred)  # (N*K, 1, H, W) in [0, π)
        direction_values = self._transform_gt_sem_seg(gt)  # (N*K, div, H, W)
        mask = self._get_loss_mask(gt)

        # Weighted circular mean of GT (in bin-angle space)
        weights = direction_values / (direction_values.sum(1, keepdim=True) + self.EPS)
        gt_sin = (weights * torch.sin(self.direction_bins)).sum(1, keepdim=True)
        gt_cos = (weights * torch.cos(self.direction_bins)).sum(1, keepdim=True)

        # Normalize to unit vector
        gt_len = torch.sqrt(gt_sin**2 + gt_cos**2 + self.EPS)
        gt_cos_unit = gt_cos / gt_len
        gt_sin_unit = gt_sin / gt_len

        # Predicted unit vector in doubled-angle space
        pred_angle_d = pred_angle * 2.0 if self.doubled_angles else pred_angle
        similarity = torch.clamp(
            torch.cos(pred_angle_d) * gt_cos_unit + torch.sin(pred_angle_d) * gt_sin_unit,
            -1.0,
            1.0,
        )
        loss = 1.0 - similarity

        return self._reduce(loss, mask, weight)
