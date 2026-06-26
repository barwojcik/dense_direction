"""
Smoothness loss class.

This module contains SmoothnessLoss.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import FUNCTIONS
from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class SmoothnessLoss(nn.Module):
    """
    SmoothnessLoss class.

    This class implements the smoothness loss. It encourages directional consistency in
    local neighborhoods by comparing each pixel's direction vector against a weighted sum
    of its neighbors, where the neighbor weights are defined by a configurable spatial kernel.

    Args:
        pad (int, optional): Pad size for kernels. It's used to calculate kernel size as
            2 * pad + 1. Default: 1.
        mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
        alpha (float, optional): The exponent parameter for smoothness loss. Default: 2.0.
        kernel_cfg (ConfigType, optional): Kernel configuration dict. The kernel function
            must accept ``pad`` as a keyword argument and return a ``(k_size, k_size)``
            weight tensor with the center pixel set to 0.
            Default: ``dict(type="uniform_smoothness_kernel")``.
        reduction (str, optional): Loss reduction method, available 'mean', 'sum', 'none'.
            Default: 'mean'.
        loss_weight (float, optional): Loss weight. Default: 1.0.
        loss_name (str, optional): Name of the loss. Default: "loss_smth".
    """

    DEFAULT_KERNEL_CFG = dict(type="uniform_smoothness_kernel")

    def __init__(
        self,
        pad: int = 1,
        mask_thr: float = 0.5,
        alpha: float = 2.0,
        kernel_cfg: ConfigType = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_smth",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SmoothnessLoss class.

        Args:
            pad (int, optional): Pad size for kernels. It's used to calculate kernel size as
                2 * pad + 1. Default: 1.
            mask_thr (float, optional): Threshold for semantic segmentation maps. Default: 0.5.
            alpha (float, optional): The exponent parameter for smoothness loss. Default: 2.0.
            kernel_cfg (ConfigType, optional): Kernel configuration dict.
                Default: ``dict(type="uniform_smoothness_kernel")``.
            reduction (str, optional): Loss reduction method, available 'mean', 'sum', 'none'.
                Default: 'mean'.
            loss_weight (float, optional): Loss weight. Default: 1.0.
            loss_name (str, optional): Name of the loss. Default: "loss_smth".
        """
        super().__init__()
        self.pad = pad
        self.k_size = 2 * pad + 1
        self.mask_thr: float = mask_thr
        self.alpha: float = alpha
        self.reduction: str = reduction.lower()
        self.loss_weight: float = loss_weight
        self._loss_name: str = loss_name

        self.kernel_cfg: ConfigType = (kernel_cfg or self.DEFAULT_KERNEL_CFG).copy()
        kernel_fn: Callable = FUNCTIONS.get(self.kernel_cfg.pop("type"))
        kernel: Tensor = kernel_fn(pad=pad, **self.kernel_cfg)  # (k_size, k_size)

        # Flatten to (1, 1, k_size^2, 1) for broadcasting over (N*K, 2, k_size^2, H*W)
        kernel_flat = kernel.reshape(1, 1, self.k_size**2, 1)
        self.register_buffer("kernel_weights", kernel_flat.float())

    def _get_loss_mask(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Computes loss mask.

        Args:
            gt_sem_seg (Tensor): Ground truth semantic segmentation map of shape (N * K, 1, H, W).

        Returns:
            Tensor: Loss mask of shape (N * K, 1, H, W).
        """
        return torch.where(gt_sem_seg > self.mask_thr, 1, 0)

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes smoothness loss.

        Args:
            pred_vector_field (Tensor): Per class 2D vector field of shape (N, K, 2, H, W).
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, 1, H, W).
            weight (float, optional): Optional weight for smoothness loss value.

        Returns:
            Tensor: Smoothness loss value.
        """
        n, k, _, h, w = pred_vector_field.shape
        pred_vector_field = pred_vector_field.reshape(-1, 2, h, w)  # (N*K, 2, H, W)
        gt_sem_seg = gt_sem_seg.reshape(-1, 1, h, w)  # (N*K, 1, H, W)

        loss_mask = self._get_loss_mask(gt_sem_seg)
        masked_vector_field = pred_vector_field * loss_mask

        # Extract all local patches: (N*K, 2*k_size^2, H*W)
        neighborhood_patches = F.unfold(masked_vector_field, self.k_size, padding=self.pad)
        # Reshape to (N*K, 2, k_size^2, H*W) for per-channel weighted summation
        neighborhood_patches = neighborhood_patches.view(n * k, 2, self.k_size**2, h * w)
        # Weighted sum over neighbors; center weight is 0, so no self-subtraction needed
        neighborhood_vectors = (neighborhood_patches * self.kernel_weights).sum(dim=2)
        neighborhood_vectors = neighborhood_vectors.reshape(n * k, 2, h, w)

        loss = F.cosine_similarity(masked_vector_field, neighborhood_vectors, dim=1, eps=1e-8)
        loss = torch.nan_to_num(loss, nan=0.0)
        loss = (0.5 * (1 - loss)) ** self.alpha
        loss = loss.unsqueeze(1) * loss_mask
        loss = loss * self.loss_weight * (weight or 1.0)

        if self.reduction == "mean":
            return loss.sum() / loss_mask.sum()

        if self.reduction == "sum":
            return loss.sum()

        return loss

    @property
    def loss_name(self) -> str:
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
