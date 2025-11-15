"""
Directional loss class.

This module contains DirectionalLoss.
"""

from typing import Callable

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmengine import FUNCTIONS
from mmseg.registry import MODELS
from mmseg.utils import ConfigType


@MODELS.register_module()
class DirectionalLoss(nn.Module):
    """
    DirectionalLoss class.

    This class implements the directional loss.

    Args:
        pad (int, optional): Pad size for kernels. Default: 3.
        div (int, optional): Division factor for direction bins. Default: 20.
        mask_thr (float, optional): Threshold for sematic segmentation maps. Default: 0.5.
        squish_values (bool, optional): Whether to squish values. Default: False.
        norm_values (bool, optional): Whether to normalize values. Default: False.
        norm_order (int, optional): Order of value normalization. Default: 1.
        mask_patches (bool, optional): Whether to mask patches. Default: False.
        patch_thr (float, optional): Threshold for patch masking. Default: 0.8.
        kernel_cfg (ConfigType, optional): Kernel configuration.
            Default: 'dict(type="circular_point_kernel")'.
        reduction (str, optional): Loss reduction method, available 'mean', 'sum', 'none'.
            Default: 'mean'.
        loss_weight (float, optional): Loss weight. Default: 1.0.
        loss_name (str, optional): Name of the loss. Default: "loss_dir".
    """

    DEFAULT_KERNEL_CFG = dict(type="circular_point_kernel")

    def __init__(
        self,
        pad: int = 3,
        div: int = 20,
        mask_thr: float = 0.5,
        squish_values: bool = False,
        norm_values: bool = False,
        norm_order: int = 1,
        mask_patches: bool = False,
        patch_thr: float = 0.8,
        kernel_cfg: ConfigType = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_dir",
        **kwargs,
    ) -> None:
        """
        Initializes the DirectionalLoss instance.

        Args:
            pad (int, optional): Pad size for kernels. Default: 3.
            div (int, optional): Division factor for direction bins. Default: 20.
            mask_thr (float, optional): Threshold for sematic segmentation maps. Default: 0.5.
            squish_values (bool, optional): Whether to squish values. Default: False.
            norm_values (bool, optional): Whether to normalize values. Default: False.
            norm_order (int, optional): Order of value normalization. Default: 1.
            mask_patches (bool, optional): Whether to mask patches. Default: False.
            patch_thr (float, optional): Threshold for patch masking. Default: 0.8
            kernel_cfg (ConfigType, optional): Kernel configuration.
                Default: 'dict(type="circular_point_kernel")'.
            reduction (str, optional): Loss reduction method, available 'mean', 'sum', 'none'.
                Default: 'mean'.
            loss_weight (float, optional): Loss weight. Default: 1.0.
            loss_name (str, optional): Name of the loss. Default: "loss_dir".
        """

        super().__init__()
        self.k_size: int = 2 * pad + 1
        self.pad: int = pad
        self.div: int = div
        self.mask_thr: float = mask_thr
        self.squish_values: bool = squish_values
        self.norm_values: bool = norm_values
        self.norm_order: int | str = norm_order
        self.mask_patches: bool = mask_patches
        self.patch_thr: float = patch_thr
        kernel_cfg: ConfigType = (kernel_cfg or self.DEFAULT_KERNEL_CFG).copy()
        self.kernel_fn: Callable = FUNCTIONS.get(kernel_cfg.pop("type"))
        self.kernel_cfg: ConfigType = kernel_cfg
        self.reduction: str = reduction.lower()
        self.loss_weight: float = loss_weight
        self._loss_name: str = loss_name

        transform_weights = self._get_kernels().unsqueeze(1)
        direction_bins = np.linspace(0, 1, div, endpoint=False)
        direction_bins = torch.tensor(direction_bins).reshape(1, div, 1, 1)

        self.register_buffer("transform_weights", transform_weights.float())
        self.register_buffer("direction_bins", direction_bins.float())
        self.register_buffer("pi", torch.tensor(np.pi).float())

    def _get_kernels(self) -> torch.Tensor:
        """
        Computes kernels for directional loss.

        Returns:
            Tensor: Kernels for directional loss.
        """

        return self.kernel_fn(
            k_size=self.k_size,
            pad=self.pad,
            div=self.div,
            **self.kernel_cfg,
        )

    def _convert_to_direction(self, predictions: Tensor) -> Tensor:
        """
        Converts 2D vector field into values in range from 0 to 1 that represent direction.

        Args:
            predictions (Tensor): Per class 2D vector field of shape (N * K, 2, H, W).

        Returns:
            Tensor: Direction as values that range from 0 to 1.
        """

        x_component, y_component = torch.unbind(predictions, dim=1)  # n * k, h, w

        # Angles in radians in range of pi to -pi
        angles = torch.atan2(y_component, x_component).unsqueeze(1)  # n * k, 1, h, w

        # Direction as a range from 0 to 1 representing range 0 to pi (0 to 180 degrees)
        direction = (angles + self.pi) / (2 * self.pi)

        return direction

    @torch.no_grad()
    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Transforms a ground truth semantic segmentation map into a map that contains direction
        values.

        Args:
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, 1, H, W).

        Returns:
            Tensor: Transformed ground truth semantic segmentation map of shape (N, K, div, H, W).
        """

        return F.conv2d(
            input=gt_sem_seg.float(),
            weight=self.transform_weights,
            padding=self.pad,
        )

    def _get_loss_mask(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Computes loss mask.

        Args:
            gt_sem_seg (Tensor): Ground truth semantic segmentation map of shape (N * K, 1, H, W).

        Returns:
            Tensor: Loss mask of shape (N * K, 1, H, W).
        """
        return torch.where(gt_sem_seg > self.mask_thr, 1, 0)

    def _get_patch_mask(self, direction_values: Tensor) -> Tensor:
        """
        Computes patch mask.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map of
                shape (N * K, div, H, W).

        Returns:
            Tensor: Patch mask of shape (N * K, 1, H, W).
        """
        patch_values = direction_values.mean(dim=1, keepdim=True)
        return torch.where(patch_values > self.patch_thr, 0, 1)

    @staticmethod
    def _squish_direction_values(direction_values: Tensor) -> Tensor:
        """
        Changes the range of direction values form 0-1 to min-1.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map of
                shape (N * K, div, H, W).

        Returns:
            Tensor: Squished direction_values.
        """
        direction_values_min = direction_values.min(dim=1, keepdim=True)[0]
        direction_values = (direction_values - direction_values_min) / (1 - direction_values_min)
        direction_values = torch.nan_to_num(direction_values, 0)
        return direction_values

    def _norm_direction_values(self, direction_values: Tensor) -> Tensor:
        """
        Normalizes direction values.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map of
                shape (N * K, div, H, W).

        Returns:
            Tensor: Normalized direction_values.
        """
        direction_values_norm = torch.linalg.vector_norm(
            direction_values, ord=self.norm_order, dim=1, keepdim=True
        )
        direction_values /= direction_values_norm
        direction_values = torch.nan_to_num(direction_values, 0)
        return direction_values

    def forward(
        self, pred_vector_field: Tensor, gt_sem_seg: Tensor, weight: float = None, **kwargs
    ) -> Tensor:
        """
        Computes directional loss.

        Args:
            pred_vector_field (Tensor): Per class 2D vector field of shape (N, K, 2, H, W).
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, 1, H, W).
            weight (float, optional): Optional weight for directional loss value.

        Returns:
            Tensor: Directional loss value.
        """
        h, w = pred_vector_field.shape[-2:]
        pred_vector_field = pred_vector_field.reshape(-1, 2, h, w)  # n * k, 2, h, w

        pred_direction = self._convert_to_direction(pred_vector_field)  # n * k, 1, h, w
        pred_direction = pred_direction.repeat(1, self.div, 1, 1)  # n * k, div, h, w

        shifted_direction_bins = (self.direction_bins - pred_direction).abs()
        direction_weights = torch.sin(shifted_direction_bins * self.pi)

        gt_sem_seg = gt_sem_seg.reshape(-1, 1, h, w)  # n * k, 1, h, w
        direction_values = self._transform_gt_sem_seg(gt_sem_seg)  # n * k, div, h, w

        loss_mask = self._get_loss_mask(gt_sem_seg)
        if self.mask_patches:
            patch_mask = self._get_patch_mask(direction_values)
            loss_mask = loss_mask * patch_mask

        if self.squish_values:
            direction_values = self._squish_direction_values(direction_values)

        if self.norm_values:
            direction_values = self._norm_direction_values(direction_values)

        loss = direction_weights * direction_values
        loss = (loss * loss_mask).sum(1, keepdim=True)
        loss = loss * self.loss_weight * (weight or 1.0)

        if self.reduction == "mean":
            return loss.sum() / loss_mask.sum()

        if self.reduction == "sum":
            return loss.sum()

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
