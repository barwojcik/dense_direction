"""
Linear Direction Decode Head Class.

A simple linear decode head class used for direction estimation tasks.
"""

from typing import Sequence

from torch import Tensor

from mmcv.cnn.bricks.norm import build_norm_layer
from mmseg.registry import MODELS

from .base_direction import BaseDirectionDecodeHead


@MODELS.register_module()
class LinearDirectionHead(BaseDirectionDecodeHead):
    """
    Linear decode head for a direction estimation task.

    Simple linear head with optional normalization layer before final prediction.

    Args:
        **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
    """

    def __init__(self, **kwargs) -> None:
        """
        Linear decode head for a direction estimation task.

        Simple linear head with optional normalization layer before final prediction.

        Args:
            **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
        """
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.with_norm: bool = self.norm_cfg is not None
        if self.with_norm:
            _, self.norm = build_norm_layer(self.norm_cfg, self.in_channels)

    def layers(self, inputs: Sequence[Tensor]) -> Tensor:
        """
        Forward pass through the linear deocde head layers.

        This method transforms inputs and passes them through the optional normalization layer.

        Args:
            inputs (Sequence[Tensor]): List of input tensors.

        Returns:
            features (Tensor): Output tensors of shape (N, C, H, W).
        """
        features: Tensor = self._transform_inputs(inputs)
        if self.with_norm:
            features = self.norm(features)

        return features

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """
        Forward pass through the decode head.

        Args:
            inputs (Sequence[Tensor]): Input tensors of shape (N, C, H, W).

        Returns:
            outputs (Tensor): Output direction vector field for each class (N, K, 2, H, W).
        """
        features: Tensor = self.layers(inputs)
        outputs: Tensor = self.estimate_directions(features)

        return outputs
