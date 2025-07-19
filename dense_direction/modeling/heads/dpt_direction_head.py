"""
Vision Transformers for Dense Prediction.

This module implements DPTDirectionHead class that is a direction estimation counterpart of
mmseg's DPTHead.
"""

from typing import Sequence

from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType

from .direction_head import BaseDirectionDecodeHead
from ..blocks import DPTDecoderBlock


@MODELS.register_module()
class DPTDirectionHead(BaseDirectionDecodeHead):
    """
    Vision Transformers for Dense Prediction.

    This class is a direction estimation counterpart of mmseg's DPTHead.

    This head is an implementation of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Output channels of post-process conv layers.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether to expand the channels in post-process block.
            Default: False.
        act_cfg (dict): The activation config for residual conv unit. Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN').
    """

    def __init__(
        self,
        embed_dims: int = 768,
        post_process_channels: Sequence[int] = None,
        readout_type: str = "ignore",
        patch_size: int = 16,
        expand_channels: bool = False,
        act_cfg: ConfigType = None,
        norm_cfg: ConfigType = None,
        dpt_init_cfg: ConfigType = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DPTDirectionHead class.

        Vision Transformers for Dense Prediction. This class is a direction estimation counterpart
        of mmseg's DPTHead.

        This head is an implementation of `DPT <https://arxiv.org/abs/2103.13413>`_.

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
            dpt_init_cfg (dict): Config dict for initializing DPT layers. Default: None.
        """

        super().__init__(**kwargs)
        self.dpt_block = DPTDecoderBlock(
            channels=self.channels,
            embed_dims=embed_dims,
            post_process_channels=post_process_channels,
            readout_type=readout_type,
            patch_size=patch_size,
            expand_channels=expand_channels,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=dpt_init_cfg,
        )

    def layers(self, inputs: list[Tensor]) -> Tensor:
        """
        Forward pass through the DPT deocde head layers.

        This method transforms inputs and passes them through head's layers.

        Args:
            inputs (list[Tensor]): List of input tensors.

        Returns:
            features (Tensor): Output tensors of shape (N, C, H, W).
        """

        feature_list: list[Tensor] = self._transform_inputs(inputs)
        features = self.dpt_block(feature_list)
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
