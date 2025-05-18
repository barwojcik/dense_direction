"""
Linear Segmentation Decode Head Class.

A simple linear decode head class used for segmentation tasks.
"""

import torch

from mmcv.cnn.bricks.norm import build_norm_layer
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class LinearHead(BaseDecodeHead):
    """
    Linear decode head for a segmentation task.

    Simple linear head with optional normalization layer before final prediction.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg. Default: None.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundled into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of the corresponding loss function which
            could be shown in the training log. If you want this loss
            item to be included in the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g., dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,**kwargs) -> None:
        """
        Linear decode head for a segmentation task.

        Simple linear head with optional normalization layer before final prediction.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            channels (int): Channels after modules, before conv_seg.
            num_classes (int): Number of classes.
            out_channels (int): Output channels of conv_seg. Default: None.
            threshold (float): Threshold for binary segmentation in the case of
                `num_classes==1`. Default: None.
            dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
            conv_cfg (dict|None): Config of conv layers. Default: None.
            norm_cfg (dict|None): Config of norm layers. Default: None.
            act_cfg (dict): Config of activation layers.
                Default: dict(type='ReLU')
            in_index (int|Sequence[int]): Input feature index. Default: -1
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
                Default: None.
            loss_decode (dict | Sequence[dict]): Config of decode loss.
                The `loss_name` is property of corresponding loss function which
                could be shown in training log. If you want this loss
                item to be included into the backward graph, `loss_` must be the
                prefix of the name. Defaults to 'loss_ce'.
                 e.g. dict(type='CrossEntropyLoss'),
                 [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
                  dict(type='DiceLoss', loss_name='loss_dice')]
                Default: dict(type='CrossEntropyLoss').
            ignore_index (int | None): The label index to be ignored. When using
                masked BCE loss, ignore_index should be set to None. Default: 255.
            sampler (dict|None): The config of segmentation map sampler.
                Default: None.
            align_corners (bool): align_corners argument of F.interpolate.
                Default: False.
            init_cfg (dict or list[dict], optional): Initialization config dict.
        """
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.with_norm: bool = self.norm_cfg is not None
        if self.with_norm:
            _, self.norm = build_norm_layer(self.norm_cfg, self.in_channels)

    def _layers(self, inputs) -> torch.Tensor:
        """
        Forward pass through the linear deocde head layers.

        This method transforms inputs and passes them through the optional normalization layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            features (torch.Tensor): Output tensor with clas logits.
        """
        features = self._transform_inputs(inputs)
        if self.with_norm:
            features = self.norm(features)

        return features

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass through the decode head.

        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            outputs (torch.Tensor): Output tensor with clas logits.
        """
        features = self._layers(inputs)
        outputs = self.cls_seg(features)

        return outputs
