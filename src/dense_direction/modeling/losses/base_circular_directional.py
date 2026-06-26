"""
Base classes for circular directional losses.

This module provides two base classes:

- ``BaseCircularDirectionalLoss``: Standard variant that uses ``F.conv2d`` to apply
  kernel weights over the full spatial map.
- ``EfficientBaseCircularDirectionalLoss``: Memory-efficient variant that uses
  ``F.unfold`` + ``F.linear`` and only processes pixels inside the segmentation mask,
  avoiding materializing the full ``(N*K, div, H, W)`` intermediate tensor.

All directional losses that operate on undirected orientations (i.e. treat a direction
and its 180° opposite as identical) should inherit from one of these classes.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import FUNCTIONS
from mmseg.utils import ConfigType
from torch import Tensor


class BaseCircularDirectionalLoss(nn.Module):
    """
    Base class for circular directional losses (standard conv2d variant).

    Provides shared infrastructure:
    - kernel loading and buffer registration
    - ``_convert_to_direction``: atan2 → angle in ``[0, π)``
    - ``_transform_gt_sem_seg``: kernel convolution → ``(N*K, div, H, W)``
    - ``_get_loss_mask``: binary mask from GT segmentation
    - ``_reduce``: applies ``loss_weight``, external ``weight``, and reduction

    Subclasses must implement ``forward``.

    Args:
        pad (int, optional): Pad size; kernel size = 2*pad+1. Default: 3.
        div (int, optional): Number of direction bins. Default: 20.
        mask_thr (float, optional): Threshold for binary GT mask. Default: 0.5.
        kernel_cfg (ConfigType, optional): Kernel config dict.
            Default: ``dict(type="circular_point_kernel")``.
        reduction (str, optional): ``'mean'``, ``'sum'``, or ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Global loss weight. Default: 1.0.
        loss_name (str, optional): Name for logging. Default: ``"loss_dir"``.
    """

    DEFAULT_KERNEL_CFG: dict = dict(type="circular_point_kernel")
    EPS: float = 1e-10

    def __init__(
        self,
        pad: int = 3,
        div: int = 20,
        mask_thr: float = 0.5,
        kernel_cfg: ConfigType = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_dir",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.k_size: int = 2 * pad + 1
        self.pad: int = pad
        self.div: int = div
        self.mask_thr: float = mask_thr
        self.kernel_cfg: ConfigType = (kernel_cfg or self.DEFAULT_KERNEL_CFG).copy()
        self.kernel_fn: Callable = FUNCTIONS.get(self.kernel_cfg.pop("type"))
        self.reduction: str = reduction.lower()
        self.loss_weight: float = loss_weight
        self._loss_name: str = loss_name

        self._setup_kernel_buffers()

    # ------------------------------------------------------------------
    # Kernel setup — overridden by the efficient subclass
    # ------------------------------------------------------------------

    def _setup_kernel_buffers(self) -> None:
        """Registers ``transform_weights`` (for conv2d) and ``pi`` as buffers."""
        kernels = self._get_kernels()  # (div, k_size, k_size)
        self.register_buffer("transform_weights", kernels.unsqueeze(1).float())
        self.register_buffer("pi", torch.tensor(np.pi).float())

    def _get_kernels(self) -> Tensor:
        """Returns direction kernels of shape ``(div, k_size, k_size)``."""
        return self.kernel_fn(
            k_size=self.k_size,
            pad=self.pad,
            div=self.div,
            **self.kernel_cfg,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _convert_to_direction(self, predictions: Tensor) -> Tensor:
        """
        Converts a 2-D vector field to undirected angles in ``[0, π)``.

        Args:
            predictions (Tensor): Shape ``(N*K, 2, H, W)``.

        Returns:
            Tensor: Angles in ``[0, π)``, shape ``(N*K, 1, H, W)``.
        """
        x, y = torch.unbind(predictions, dim=1)
        return (torch.atan2(y, x) % self.pi).unsqueeze(1)

    @torch.no_grad()
    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Applies direction kernels to the GT map via convolution.

        Args:
            gt_sem_seg (Tensor): Shape ``(N*K, 1, H, W)``.

        Returns:
            Tensor: Direction values, shape ``(N*K, div, H, W)``.
        """
        return F.conv2d(gt_sem_seg.float(), self.transform_weights, padding=self.pad)

    def _get_loss_mask(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Builds a binary mask over foreground pixels.

        Args:
            gt_sem_seg (Tensor): Shape ``(N*K, 1, H, W)``.

        Returns:
            Tensor: Integer mask ``{0, 1}``, shape ``(N*K, 1, H, W)``.
        """
        return torch.where(gt_sem_seg > self.mask_thr, 1, 0)

    def _reduce(
        self,
        loss: Tensor,
        mask: Tensor,
        weight: float | None = None,
    ) -> Tensor:
        """
        Applies mask, scales by ``loss_weight`` and optional ``weight``, then reduces.

        Args:
            loss (Tensor): Per-pixel loss, shape ``(N*K, 1, H, W)``.
            mask (Tensor): Binary mask, shape ``(N*K, 1, H, W)``.
            weight (float | None): Optional additional scalar weight.

        Returns:
            Tensor: Scalar loss (or spatial map when reduction is ``'none'``).
        """
        loss = loss * mask * self.loss_weight * (weight or 1.0)
        if self.reduction == "mean":
            return loss.sum() / (mask.sum() + self.EPS)
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_vector_field: Tensor,
        gt_sem_seg: Tensor,
        weight: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError

    @property
    def loss_name(self) -> str:
        """Loss Name."""
        return self._loss_name


# ---------------------------------------------------------------------------


class EfficientBaseCircularDirectionalLoss(BaseCircularDirectionalLoss):
    """
    Memory-efficient base class for circular directional losses.

    Instead of materializing the full ``(N*K, div, H, W)`` direction-value tensor,
    this variant uses ``F.unfold`` + ``F.linear`` and only processes the pixels
    that fall inside the foreground mask.  This reduces peak memory usage when
    training on large images or with many direction bins.

    The API is identical to ``BaseCircularDirectionalLoss``.  Subclasses implement
    ``forward`` using the helpers ``_convert_to_direction_flat``,
    ``_get_mask_idx``, and ``_transform_gt_sem_seg_efficient``.
    """

    # ------------------------------------------------------------------
    # Override kernel setup — flat (div, k_size^2) for F.linear
    # ------------------------------------------------------------------

    def _setup_kernel_buffers(self) -> None:
        """Registers ``transform_weights`` (for F.linear) and ``pi`` as buffers."""
        kernels = self._get_kernels()  # (div, k_size, k_size)
        self.register_buffer("transform_weights", kernels.reshape(self.div, -1).float())
        self.register_buffer("pi", torch.tensor(np.pi).float())

    # ------------------------------------------------------------------
    # Efficient helpers — operate on flat masked positions
    # ------------------------------------------------------------------

    def _convert_to_direction_flat(self, pred_vector_field: Tensor) -> Tensor:
        """
        Converts a 2-D vector field to flat undirected angles in ``[0, π)``.

        Args:
            pred_vector_field (Tensor): Shape ``(N, K, 2, H, W)``.

        Returns:
            Tensor: Flat angles in ``[0, π)``, shape ``(N*K*H*W,)``.
        """
        x, y = torch.unbind(pred_vector_field, dim=2)  # (N, K, H, W)
        return (torch.atan2(y, x) % self.pi).reshape(-1)

    def _get_mask_idx(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Returns flat indices of foreground pixels.

        Args:
            gt_sem_seg (Tensor): Shape ``(N, K, H, W)``.

        Returns:
            Tensor: 1-D index tensor of foreground positions.
        """
        return torch.where(gt_sem_seg > self.mask_thr, 1, 0).view(-1).nonzero().squeeze(1)

    @torch.no_grad()
    def _transform_gt_sem_seg_efficient(
        self,
        gt_sem_seg: Tensor,
        mask_idx: Tensor,
    ) -> Tensor:
        """
        Applies direction kernels only to masked pixels via unfold + linear.

        Args:
            gt_sem_seg (Tensor): Shape ``(N, K, H, W)`` (channel dim already squeezed).
            mask_idx (Tensor): Flat indices from ``_get_mask_idx``.

        Returns:
            Tensor: Direction values for masked pixels, shape ``(M, div)``.
        """
        n, k, h, w = gt_sem_seg.shape
        patches = F.unfold(
            gt_sem_seg.reshape(n * k, 1, h, w).float(),
            self.k_size,
            padding=self.pad,
        )  # (N*K, k_size^2, H*W)
        patches = patches.permute(0, 2, 1).reshape(-1, self.k_size**2)  # (N*K*H*W, k_size^2)
        filtered = patches.index_select(0, mask_idx)  # (M, k_size^2)
        return F.linear(filtered, self.transform_weights)  # (M, div)

    def _reduce_flat(
        self,
        loss: Tensor,
        weight: float | None = None,
    ) -> Tensor:
        """
        Reduces a flat per-pixel loss tensor ``(M,)``.

        Args:
            loss (Tensor): Per-pixel losses for masked positions, shape ``(M,)``.
            weight (float | None): Optional additional scalar weight.

        Returns:
            Tensor: Scalar loss (or 1-D tensor when reduction is ``'none'``).
        """
        if loss.numel() == 0:
            return loss.sum()  # zero with gradient
        loss = loss * self.loss_weight * (weight or 1.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
