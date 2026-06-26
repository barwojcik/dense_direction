"""
Von Mises Directional Loss class.

This module contains VonMisesDirectionalLoss, a directional loss based on the negative
log-likelihood of the von Mises distribution. It uses the doubled-angle trick to handle
the 180° periodicity of undirected line directions.
"""

from typing import Any

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import Tensor

from .base_circular_directional import BaseCircularDirectionalLoss


@MODELS.register_module()
class VonMisesDirectionalLoss(BaseCircularDirectionalLoss):
    """
    VonMisesDirectionalLoss class.

    Directional loss based on the von Mises NLL. The predicted angle and the GT circular
    mean angle (derived from kernel direction values) are both doubled before computing
    the cosine loss, which maps the 180° periodicity of undirected directions onto a full
    360° circle and makes the loss properly periodic.

    Loss = (-κ·cos(2θ_pred - 2θ_gt) + κ) / (2κ)   (normalized to [0, 1])

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
        self.register_buffer("direction_bins", bins.reshape(1, self.div, 1, 1).float())

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes Von Mises directional loss.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.
            gt_sem_seg (Tensor): Shape ``(N, K, 1, H, W)``.
            weight (float | None): Optional external loss weight.

        Returns:
            Tensor: Von Mises directional loss value.
        """
        h, w = pred_vector_field.shape[-2:]
        pred = pred_vector_field.reshape(-1, 2, h, w)
        gt = gt_sem_seg.reshape(-1, 1, h, w)

        pred_angle = self._convert_to_direction(pred)  # (N*K, 1, H, W) in [0, π)
        direction_values = self._transform_gt_sem_seg(gt)  # (N*K, div, H, W)
        mask = self._get_loss_mask(gt)

        # Weighted circular mean of GT angles in bin space
        weights = direction_values / (direction_values.sum(1, keepdim=True) + self.EPS)
        gt_sin = (weights * torch.sin(self.direction_bins)).sum(1, keepdim=True)
        gt_cos = (weights * torch.cos(self.direction_bins)).sum(1, keepdim=True)
        gt_angle = torch.atan2(gt_sin, gt_cos)  # (N*K, 1, H, W) in [-π, π]

        pred_angle_d = pred_angle * 2.0 if self.doubled_angles else pred_angle
        delta = pred_angle_d - gt_angle
        loss = (-self.kappa * torch.cos(delta) + self.kappa) / (2 * self.kappa)

        return self._reduce(loss, mask, weight)
