"""
Dual Decode Head class for simultaneous segmentation and direction estimation.

This module provides a DualDecodeHead class, this class is a wrapper that wraps segmentation and
direction heads.
"""

from typing import Sequence

from torch import Tensor
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class DualDecodeHead(nn.Module):
    """
    DualDecodeHead for simultaneous segmentation and direction estimation.

    This class is a wrapper that wraps segmentation and direction heads.

    Args:
         seg_head_config (ConfigType): The config for the segmentation head.
         dir_head_config (ConfigType): The config for the direction head.
    """

    def __init__(
        self,
        seg_head_config: ConfigType,
        dir_head_config: ConfigType,
    ) -> None:
        super().__init__()
        self.seg_head = MODELS.build(seg_head_config)
        self.dir_head = MODELS.build(dir_head_config)

        self.align_corners = self.seg_head.align_corners
        self.num_classes = self.seg_head.num_classes
        self.out_channels = self.seg_head.out_channels
        self.dir_classes = self.dir_head.dir_classes

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the decode heads.

        Args:
            inputs (Sequence[Tensor]): Input tensors of shape (N, C, H, W).

        Returns:
            outputs (tuple[Tensor, Tensor]): Output tensors for respective heads.
        """
        seg_logits = self.seg_head.forward(inputs)
        dir_vector_field = self.dir_head.forward(inputs)

        return seg_logits, dir_vector_field

    def loss(
        self, inputs: tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType
    ) -> dict:
        """
        Forward function for training.

        This method wraps segmentation head and direction head loss calls and returns merged loss
        dictionary.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually
                includes information such as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_losses: dict = self.seg_head.loss(inputs, batch_data_samples, train_cfg)
        seg_losses = {f"seg_head_{key}": value for key, value in seg_losses.items()}

        dir_losses: dict = self.dir_head.loss(inputs, batch_data_samples, train_cfg)
        dir_losses = {f"dir_head_{key}": value for key, value in dir_losses.items()}

        return seg_losses | dir_losses

    def predict(
        self, inputs: tuple[Tensor], batch_img_metas: list[dict], test_cfg: ConfigType
    ) -> tuple[Tensor, Tensor]:
        """
        Forward function for prediction.

        Args:
            inputs (tuple[Tensor]): List of multi-level img features.
            batch_img_metas (list[dict]): List Image info where each dict may also contain: 'img_shape',
                'scale_factor', 'flip', 'img_path', 'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            tuple[Tensor, Tensor]: Output tensors for respective heads.
        """
        seg_logits = self.seg_head.predict(inputs, batch_img_metas, test_cfg)
        dir_vector_field = self.dir_head.predict(inputs, batch_img_metas, test_cfg)

        return seg_logits, dir_vector_field
