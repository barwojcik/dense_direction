"""
Efficient directional loss class.

This module contains EfficientDirectionalLoss, an alternative implementation of DirectionalLoss
with reduced memory footprint.
"""

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
        mask_thr (float, optional): Threshold for masking pixels. Default: 0.5.
        kernel_thr (float, optional): Threshold for kernel computation. Default: 0.8.
        kernel_cfg (ConfigType, optional): Kernel configuration.
            Default: 'dict(type="circular_point_kernels")'.
        loss_name (str, optional): Name of the loss. Default: "loss_dir".
    """


    DEFAULT_KERNEL_CFG = dict(type="circular_point_kernels")

    def __init__(
        self,
        pad=3,
        div=20,
        mask_thr=0.5,
        kernel_thr=0.8,
        kernel_cfg: ConfigType = None,
        loss_name="loss_dir",
        **kwargs,
    ) -> None:
        super().__init__()
        """
        Initializes the DirectionalLoss instance.

        Args:
            pad (int, optional): Pad size for kernels. Default: 3.
            div (int, optional): Division factor for direction bins. Default: 20.
            mask_thr (float, optional): Threshold for masking gt_sem_seg. Default: 0.5.
            kernel_thr (float, optional): Threshold for kernel computation. Default: 0.8.
            kernel_cfg (ConfigType, optional): Kernel function configuration. 
                Default: 'dict(type="circular_point_kernels")'.
            loss_name (str, optional): Name of the loss. Default: "loss_dir".
        """

        self.k_size = 2 * pad + 1
        self.pad = pad
        self.div = div
        self.mask_thr = mask_thr
        self.kernel_thr = kernel_thr
        kernel_cfg = kernel_cfg or self.DEFAULT_KERNEL_CFG
        self.kernel_fn = FUNCTIONS.get(kernel_cfg.pop("type"))
        self.kernel_cfg = kernel_cfg
        self._loss_name = loss_name

        transform_weights = self._get_kernels().float()
        direction_bins = np.linspace(0, 1, div, endpoint=False)
        direction_bins = torch.tensor(direction_bins).float().reshape(1, div)

        self.register_buffer("transform_weights", transform_weights)
        self.register_buffer("direction_bins", direction_bins)
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
            mask_thr=self.mask_thr,
            kernel_thr=self.kernel_thr,
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

        x_component = predictions[:, :, 0, ...]  # n, k, h, w
        y_component = predictions[:, :, 1, ...]  # n, k, h, w

        # Angles in radians in range of pi to -pi
        angles = torch.atan2(y_component, x_component).unsqueeze(2)  # n, k, 1, h, w

        # Direction as a range from 0 to 1 representing range 0 to pi (0 to 180 degrees)
        direction = (angles + self.pi) / (2 * self.pi)

        return direction

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
            Tensor: Transformed ground truth semantic segmentation map of shape (len(kernel_mask),
                div).
        """

        unfolded = F.unfold(gt_sem_seg, self.k_size, padding=self.pad)  # n, k_size^2, k * h * w
        unfolded = unfolded.permute(0, 2, 1)  # n, k * h * w, k_size^2
        unfolded = unfolded.reshape(-1, self.k_size ** 2)  # n * k * h * w, k_size^2

        filtered_unfolded = unfolded.index_select(dim=0, index=kernel_mask)

        return F.linear(
            input=filtered_unfolded,
            weight=self.transform_weights,
        )

    def forward(self, pred_vector_field: Tensor, gt_sem_seg: Tensor, **kwargs) -> Tensor:
        """
        Computes directional loss.

        Args:
            pred_vector_field (Tensor): Per class 2D vector field of shape (N, K, 2, H, W).
            gt_sem_seg (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, 1, H, W).

        Returns:
            Tensor: Directional loss value.
        """

        n, k, _, h, w = gt_sem_seg.shape
        gt_sem_seg = gt_sem_seg.squeeze(2)  # n, k, 1, h, w -> n, k, h, w

        kernel_mask = gt_sem_seg.view(-1).nonzero().squeeze()

        direction_values = self._transform_gt_sem_seg(gt_sem_seg, kernel_mask)

        pred_direction = self._convert_to_direction(pred_vector_field)  # n, k, 1, h, w
        pred_direction = pred_direction.view(-1)  # n * k * h * w

        filtered_pred = pred_direction.index_select(dim=0, index=kernel_mask)
        filtered_pred = filtered_pred.unsqueeze(-1).repeat(1, self.div)

        shifted_direction_bins = (self.direction_bins - filtered_pred).abs()
        direction_weights = torch.sin(shifted_direction_bins * self.pi)

        loss = direction_weights * direction_values
        loss = loss.mean()

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
