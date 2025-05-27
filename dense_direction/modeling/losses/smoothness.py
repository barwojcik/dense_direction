"""
Smoothness loss class.

This module contains SmoothnessLoss.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class SmoothnessLoss(nn.Module):
    """
    SmoothnessLoss class.

    This class implements the smoothness loss.

    Args:
        pad (int, optional): Pad size for kernels. It's used to calculate kernel size as
                2 * pad + 1, default: 3.
        mask_thr (float, optional): Threshold for sematic segmentation maps. Default: 0.5.
        alpha (float, optional): The exponent parameter for smoothness loss. Default: 2.0.
        loss_weight (float, optional): Loss weight. Default: 1.0.
        loss_name (str, optional): Name of the loss. Default: "loss_dir".
    """

    def __init__(
        self,
        pad: int = 1,
        mask_thr: float = 0.5,
        alpha: float = 2.0,
        loss_weight: float = 1.0,
        loss_name="loss_smth",
        **kwargs,
    ) -> None:
        """
        Initializes the SmoothnessLoss class.

        Args:
            pad (int, optional): Pad size for kernels. It's used to calculate kernel size as
                2 * pad + 1, default: 3.
            mask_thr (float, optional): Threshold for sematic segmentation maps. Default: 0.5.
            alpha (float, optional): The exponent parameter for smoothness loss. Default: 2.0.
            loss_weight (float, optional): Loss weight. Default: 1.0.
            loss_name (str, optional): Name of the loss. Default: "loss_dir".
        """

        super().__init__()
        self.pad = pad
        self.k_size = 2 * pad + 1
        self.mask_thr: float = mask_thr
        self.alpha: float = alpha
        self.loss_weight: float = loss_weight
        self._loss_name: str = loss_name

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
        self, pred_vector_field: Tensor, gt_sem_seg: Tensor, weight: float = 1., **kwargs
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
        pred_vector_field = pred_vector_field.reshape(-1, 2, h, w)  # n * k, 2, h, w
        gt_sem_seg = gt_sem_seg.reshape(-1, 1, h, w)  # n * k, 1, h, w

        loss_mask = self._get_loss_mask(gt_sem_seg)
        masked_vector_field = pred_vector_field * loss_mask

        neighborhood_patches = F.unfold(masked_vector_field, self.k_size, padding=self.pad)
        neighborhood_vectors = neighborhood_patches.view(n * k, 2, self.k_size**2, -1).sum(dim=2)
        neighborhood_vectors = neighborhood_vectors.reshape(n * k, 2, h, w) - masked_vector_field

        loss = F.cosine_similarity(masked_vector_field, neighborhood_vectors, dim=1)
        loss = (0.5 * (1 - loss))**self.alpha
        loss = loss.unsqueeze(1) * loss_mask
        loss = loss.sum() / loss_mask.sum()

        return loss * self.loss_weight * weight

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
