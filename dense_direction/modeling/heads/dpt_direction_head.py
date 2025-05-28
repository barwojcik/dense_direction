"""
Vision Transformers for Dense Prediction.

This module implements DPTDirectionHead class that is a direction estimation counterpart of
mmseg's DPTHead.
"""

import math
from typing import Sequence

from torch import Tensor
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.dpt_head import (
    ReassembleBlocks,
    FeatureFusionBlock,
)
from mmseg.utils import ConfigType

from .direction_head import BaseDirectionDecodeHead


@HEADS.register_module()
class DPTDirectionHead(BaseDirectionDecodeHead):
    """
    Vision Transformers for Dense Prediction.

    This class is a direction estimation counterpart of mmseg's DPTHead.

    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Output channels of post-process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether to expand the channels in post-process
            block. Default: False.
        act_cfg (dict): The activation config for residual conv unit.
            Default dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    DEFAULT_PP_CHANNELS = [96, 192, 384, 768]
    DEFAULT_ACT_CFG: dict = dict(type="ReLU")
    DEFAULT_NORM_CFG: dict = dict(type="BN")

    def __init__(
        self,
        embed_dims: int = 768,
        post_process_channels: Sequence[int] = None,
        readout_type: str = "ignore",
        patch_size: int = 16,
        expand_channels: bool = False,
        act_cfg: ConfigType = None,
        norm_cfg: ConfigType = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DPTDirectionHead class.

        Vision Transformers for Dense Prediction. This class is a direction estimation counterpart
        of mmseg's DPTHead.

        This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.

        Args:
            embed_dims (int): The embed dimension of the ViT backbone.
                Default: 768.
            post_process_channels (List): Output channels of post-process conv
                layers. Default: [96, 192, 384, 768].
            readout_type (str): Type of readout operation. Default: 'ignore'.
            patch_size (int): The patch size. Default: 16.
            expand_channels (bool): Whether to expand the channels in the post-process
                block. Default: False.
            act_cfg (dict): The activation config for residual conv unit.
                Default dict(type='ReLU').
            norm_cfg (dict): Config dict for normalization layer.
                Default: dict(type='BN').
        """

        super().__init__(**kwargs)
        post_process_channels = post_process_channels or self.DEFAULT_PP_CHANNELS

        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(
            embed_dims, post_process_channels, readout_type, patch_size
        )

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False
                )
            )
        self.fusion_blocks = nn.ModuleList()
        act_cfg = act_cfg or self.DEFAULT_ACT_CFG
        norm_cfg = norm_cfg or self.DEFAULT_NORM_CFG
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, act_cfg, norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=norm_cfg
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

    def layers(self, inputs: list[Tensor]) -> Tensor:
        """
        Forward pass through the DPT deocde head layers.

        This method transforms inputs and passes them through head's layers.

        Args:
            inputs (list[Tensor]): List of input tensors.

        Returns:
            features (Tensor): Output tensors of shape (N, C, H, W).
        """
        assert len(inputs) == self.num_reassemble_blocks
        feature_list: list[Tensor] = self._transform_inputs(inputs)
        feature_list = self.reassemble_blocks(feature_list)
        feature_list = [self.convs[i](feature) for i, feature in enumerate(feature_list)]
        features: Tensor = self.fusion_blocks[0](feature_list[-1])
        for i in range(1, len(self.fusion_blocks)):
            features = self.fusion_blocks[i](features, feature_list[-(i + 1)])
        features = self.project(features)
        return features

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """
        Forward pass through the decode head.

        Args:
            inputs (list[Tensor]): Input tensors of shape (N, C, H, W).

        Returns:
            outputs (Tensor): Output direction vector field for each class (N, K, 2, H, W).
        """
        features: Tensor = self.layers(inputs)
        outputs: Tensor = self.estimate_directions(features)
        return outputs
