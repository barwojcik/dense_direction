"""
KL Divergence Directional Loss class.

This module contains KLDirectionalLoss, which computes the forward KL divergence between
a soft target distribution (kernel direction values via temperature-scaled softmax) and
an approximate prediction distribution (discretized von Mises centered at the predicted
angle). Doubled angles are used for proper 180° periodicity.
"""

from typing import Any

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import Tensor

from .base_circular_directional import BaseCircularDirectionalLoss


@MODELS.register_module()
class KLDirectionalLoss(BaseCircularDirectionalLoss):
    """
    KLDirectionalLoss class.

    Computes forward KL( p ‖ q ) where:

    - **p** = temperature-scaled softmax of the kernel direction values (target)
    - **q** = discretized von Mises distribution centered at the predicted angle

    Both distributions span ``div`` bins in ``[0, 2π)`` (doubled space) so that the
    180° periodicity of undirected orientations is handled correctly.

    Args:
        pad (int, optional): Pad size for kernels. Default: 3.
        div (int, optional): Number of direction bins. Default: 20.
        mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
        temp (float, optional): Temperature for softmax of target distribution.
            Lower values sharpen the target. Default: 0.12.
        kappa_q (float, optional): Concentration of the von Mises prediction distribution.
            Default: 12.0.
        doubled_angles (bool, optional): Use doubled angles for 180° periodicity. Default: True.
        kernel_cfg (ConfigType, optional): Kernel config.
            Default: ``dict(type="circular_point_kernel")``.
        reduction (str, optional): ``'mean'``, ``'sum'``, or ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Global loss weight. Default: 1.0.
        loss_name (str, optional): Name for logging. Default: ``"loss_kl_dir"``.
    """

    def __init__(
        self,
        temp: float = 0.12,
        kappa_q: float = 12.0,
        doubled_angles: bool = True,
        loss_name: str = "loss_kl_dir",
        **kwargs: Any,
    ) -> None:
        super().__init__(loss_name=loss_name, **kwargs)
        self.temp: float = temp
        self.kappa_q: float = kappa_q
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
        Computes KL divergence directional loss.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.
            gt_sem_seg (Tensor): Shape ``(N, K, 1, H, W)``.
            weight (float | None): Optional external loss weight.

        Returns:
            Tensor: KL directional loss value.
        """
        h, w = pred_vector_field.shape[-2:]
        pred = pred_vector_field.reshape(-1, 2, h, w)
        gt = gt_sem_seg.reshape(-1, 1, h, w)

        pred_angle = self._convert_to_direction(pred)  # (N*K, 1, H, W) in [0, π)
        direction_values = self._transform_gt_sem_seg(gt)  # (N*K, div, H, W)
        mask = self._get_loss_mask(gt)

        # Target p: temperature-scaled log-softmax of direction_values
        log_p = torch.log(direction_values + self.EPS) / self.temp
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)

        # Prediction q: discretized von Mises centered at pred_angle
        pred_angle_d = pred_angle * 2.0 if self.doubled_angles else pred_angle
        log_q_unnorm = self.kappa_q * torch.cos(self.direction_bins - pred_angle_d)
        log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=1, keepdim=True)

        # Forward KL: sum_bins p * (log p - log q)
        kl = (torch.exp(log_p) * (log_p - log_q)).sum(dim=1, keepdim=True)

        return self._reduce(kl, mask, weight)
