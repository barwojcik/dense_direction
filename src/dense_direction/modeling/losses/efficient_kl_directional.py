"""
Efficient KL Divergence Directional Loss class.

Memory-efficient variant of KLDirectionalLoss that uses F.unfold + F.linear
and only processes foreground pixels, avoiding the full (N*K, div, H, W) intermediate.
"""

from typing import Any

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import Tensor

from .base_circular_directional import EfficientBaseCircularDirectionalLoss


@MODELS.register_module()
class EfficientKLDirectionalLoss(EfficientBaseCircularDirectionalLoss):
    """
    EfficientKLDirectionalLoss class.

    Memory-efficient variant of ``KLDirectionalLoss``.  Uses ``F.unfold`` +
    ``F.linear`` and only computes the KL divergence for foreground pixels.

    See ``KLDirectionalLoss`` for the mathematical definition.

    Args:
        pad (int, optional): Pad size for kernels. Default: 3.
        div (int, optional): Number of direction bins. Default: 20.
        mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
        temp (float, optional): Temperature for softmax of target distribution. Default: 0.12.
        kappa_q (float, optional): Concentration of prediction distribution. Default: 12.0.
        doubled_angles (bool, optional): Double angles for 180° periodicity. Default: True.
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
        self.register_buffer("direction_bins", bins.reshape(1, self.div).float())

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes efficient KL divergence directional loss.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.
            gt_sem_seg (Tensor): Shape ``(N, K, 1, H, W)``.
            weight (float | None): Optional external loss weight.

        Returns:
            Tensor: KL directional loss value.
        """
        pred_flat = self._convert_to_direction_flat(pred_vector_field)  # (N*K*H*W,)
        gt_sq = gt_sem_seg.squeeze(2)  # (N, K, H, W)
        mask_idx = self._get_mask_idx(gt_sq)  # (M,)

        if mask_idx.numel() == 0:
            return pred_vector_field.sum() * 0.0

        direction_values = self._transform_gt_sem_seg_efficient(gt_sq, mask_idx)  # (M, div)
        filtered_pred = pred_flat.index_select(0, mask_idx)  # (M,)

        # Target p: temperature-scaled log-softmax of direction values
        log_p = torch.log(direction_values + self.EPS) / self.temp
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)

        # Prediction q: discretized von Mises centered at filtered_pred
        pred_angle_d = filtered_pred * 2.0 if self.doubled_angles else filtered_pred
        log_q_unnorm = self.kappa_q * torch.cos(self.direction_bins - pred_angle_d.unsqueeze(-1))
        log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)

        # Forward KL: sum_bins p * (log p - log q)
        kl = (torch.exp(log_p) * (log_p - log_q)).sum(-1)  # (M,)

        return self._reduce_flat(kl, weight)
