"""
Vision Transformers for Dense Prediction.

This module implements DPTDecoderBlock class that is a DPT decoder module extracted from mmseg's
DPTHead.
"""

import math
from typing import Sequence

from torch import Tensor
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.models.decode_heads.dpt_head import (
    ReassembleBlocks,
    FeatureFusionBlock,
)
from mmseg.registry import MODELS
from mmseg.utils import ConfigType


@MODELS.register_module()
class DPTDecoderBlock(BaseModule):
    """
    Vision Transformers for Dense Prediction decoder block.

    This class is a DPT decoder module extracted from mmseg's DPTHead.

    This block is an implementation of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Output channels of post-process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether to expand the channels in the post-process block.
            Default: False.
        act_cfg (dict): The activation config for residual conv unit.
            Default dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN').
        init_cfg (dict): Config dict for initializing DPT layers. Default: None.
        return_list (bool): Whether to return a tensor packed in a list or just a tensor. Default: False.
    """

    DEFAULT_PP_CHANNELS = [96, 192, 384, 768]
    DEFAULT_ACT_CFG: dict = dict(type="ReLU")
    DEFAULT_NORM_CFG: dict = dict(type="BN")

    def __init__(
        self,
        channels: int = 768,
        embed_dims: int = 768,
        post_process_channels: Sequence[int] = None,
        readout_type: str = "ignore",
        patch_size: int = 16,
        expand_channels: bool = False,
        act_cfg: ConfigType = None,
        norm_cfg: ConfigType = None,
        init_cfg: ConfigType = None,
        return_list: bool = False,
    ) -> None:
        """
        Initialize the DPTBlock class.

        Vision Transformers for Dense Prediction decoder block. This class is a DPT decoder module
        extracted from mmseg's DPTHead.

        This block is an implementation of `DPT <https://arxiv.org/abs/2103.13413>`_.

        Args:
            channels (int): The number of output channels.
            embed_dims (int): The embed dimension of the ViT backbone.
                Default: 768.
            post_process_channels (List): Output channels of post-process conv layers.
                Default: [96, 192, 384, 768].
            readout_type (str): Type of readout operation. Default: 'ignore'.
            patch_size (int): The patch size. Default: 16.
            expand_channels (bool): Whether to expand the channels in the post-process block.
                Default: False.
            act_cfg (dict): The activation config for residual conv unit. Default dict(type='ReLU').
            norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN').
            init_cfg (dict): Config dict for initializing DPT layers. Default: None.
            return_list (bool): Whether to return a tensor packed in a list or just a tensor.
                Default: False.
        """
        super().__init__(init_cfg)
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
                    channel, channels, kernel_size=3, padding=1, act_cfg=None, bias=False
                )
            )
        self.fusion_blocks = nn.ModuleList()
        act_cfg = act_cfg or self.DEFAULT_ACT_CFG
        norm_cfg = norm_cfg or self.DEFAULT_NORM_CFG
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(channels, act_cfg, norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            channels, channels, kernel_size=3, padding=1, norm_cfg=norm_cfg
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

        self.return_list = return_list

    def forward(self, feature_list: list[Tensor]) -> Tensor | list[Tensor]:
        """
        Forward pass through the dpt decoder block.

        Args:
            feature_list (list[Tensor]): List of input tensors of shape (N, in_C, in_H, in_W).

        Returns:
            features (Tensor | list[Tensor]): Output features of shape (N, out_C, out_H, out_W) as
                a tensor or a tensor packed in a list.
        """

        assert len(feature_list) == self.num_reassemble_blocks

        feature_list = self.reassemble_blocks(feature_list)
        feature_list = [self.convs[i](feature) for i, feature in enumerate(feature_list)]

        features: Tensor = self.fusion_blocks[0](feature_list[-1])

        for i in range(1, len(self.fusion_blocks)):
            features = self.fusion_blocks[i](features, feature_list[-(i + 1)])

        features = self.project(features)

        if self.return_list:
            return [features, ]
        return features
