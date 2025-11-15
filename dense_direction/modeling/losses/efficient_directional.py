"""
Efficient directional loss class.

This module contains EfficientDirectionalLoss, an alternative implementation of DirectionalLoss
with reduced memory footprint.
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
class EfficientDirectionalLoss(nn.Module):
    """
    EfficientDirectionalLoss class.

    This class implements the efficient directional loss, a direction loss with reduced memory
    footprint.

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
            patch_thr (float, optional): Threshold for patch masking. Default: 0.8.
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

        transform_weights = self._get_kernels()
        direction_bins = np.linspace(0, 1, div, endpoint=False)
        direction_bins = torch.tensor(direction_bins).reshape(1, div)

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
        ).reshape(self.div, -1)

    def _convert_to_direction(self, predictions: Tensor) -> Tensor:
        """
        Converts 2D vector field into values in range from 0 to 1 that represent direction.

        Args:
            predictions (Tensor): Per class 2D vector field of shape (N, K, 2, H, W).

        Returns:
            Tensor: Direction as values that range from 0 to 1.
        """

        x_component, y_component = torch.unbind(predictions, dim=2)  # n * k, h, w

        # Angles in radians in range of pi to -pi
        angles = torch.atan2(y_component, x_component).unsqueeze(2)  # n, k, 1, h, w

        # Direction as a range from 0 to 1 representing range 0 to pi (0 to 180 degrees)
        direction = (angles + self.pi) / (2 * self.pi)

        return direction

    def _get_loss_mask(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Extracts indexes of a semantic segmentation map that meet the condition.

        Args:
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, H, W).

        Returns:
            Tensor: Indexes of gt_sem_seg eligible for loss calculation.
        """
        return torch.where(gt_sem_seg > self.mask_thr, 1, 0).view(-1).nonzero().squeeze()

    @torch.no_grad()
    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor, kernel_mask) -> Tensor:
        """
        Transforms a ground truth semantic segmentation map into a map that contains direction
        values.

        Args:
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, H, W).
            kernel_mask (Tensor): Mask containing filtering indexes of patches meant for
                transformation.

        Returns:
            Tensor: Tensor containing transformed ground truth semantic segmentation map of shape
                (len(kernel_mask), div).
        """

        map_patches = F.unfold(gt_sem_seg, self.k_size, padding=self.pad)  # n, k_size^2, k * h * w
        map_patches = map_patches.permute(0, 2, 1)  # n, k * h * w, k_size^2
        map_patches = map_patches.reshape(-1, self.k_size**2)  # n * k * h * w, k_size^2

        filtered_map_patches = map_patches.index_select(dim=0, index=kernel_mask)

        return F.linear(
            input=filtered_map_patches,
            weight=self.transform_weights,
        )

    @staticmethod
    def _squish_direction_values(direction_values: Tensor) -> Tensor:
        """
        Changes the range of direction values form 0-1 to min-1.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map.

        Returns:
            Tensor: Squished direction_values.
        """
        direction_values_min = direction_values.min(dim=-1, keepdim=True)[0]
        direction_values = (direction_values - direction_values_min) / (1 - direction_values_min)
        return direction_values

    def _norm_direction_values(self, direction_values: Tensor) -> Tensor:
        """
        Normalizes direction values.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map.

        Returns:
            Tensor: Normalized direction_values.
        """
        direction_values_norm = torch.linalg.vector_norm(
            direction_values, ord=self.norm_order, dim=-1, keepdim=True
        )
        direction_values /= direction_values_norm
        return direction_values

    def _get_patch_mask(self, direction_values: Tensor) -> Tensor:
        """
        Extracts indexes of patches that meet the condition.

        Args:
            direction_values (Tensor): Transformed ground truth semantic segmentation map.

        Returns:
            Tensor: Indexes of patch_values eligible for loss calculation.
        """
        patch_values = direction_values.mean(dim=-1)
        return torch.where(patch_values > self.patch_thr, 0, 1).nonzero().squeeze()

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

        pred_direction = self._convert_to_direction(pred_vector_field)  # n, k, 1, h, w
        pred_direction = pred_direction.view(-1)  # n * k * h * w

        gt_sem_seg = gt_sem_seg.squeeze(2)  # n, k, 1, h, w -> n, k, h, w
        loss_mask = self._get_loss_mask(gt_sem_seg)

        direction_values = self._transform_gt_sem_seg(gt_sem_seg, loss_mask)
        filtered_pred = pred_direction.index_select(dim=0, index=loss_mask)

        if self.mask_patches:
            patch_mask = self._get_patch_mask(direction_values)
            direction_values = direction_values.index_select(dim=0, index=patch_mask)
            filtered_pred = filtered_pred.index_select(dim=0, index=patch_mask)

        if self.squish_values:
            direction_values = self._squish_direction_values(direction_values)

        if self.norm_values:
            direction_values = self._norm_direction_values(direction_values)

        filtered_pred = filtered_pred.unsqueeze(-1).repeat(1, self.div)
        shifted_direction_bins = (self.direction_bins - filtered_pred).abs()
        direction_weights = torch.sin(shifted_direction_bins * self.pi)

        loss = direction_weights * direction_values
        loss = loss.sum(-1) * self.loss_weight * (weight or 1.0)

        if self.reduction == "mean":
            return loss.mean()

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
