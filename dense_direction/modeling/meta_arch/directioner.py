"""
Directioner meta architecture.

This module provides the Directioner class, a modification to EncoderDecoder class that adopts it
to direction estimation.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from mmengine.structures import PixelData
from mmseg.models import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (
    ConfigType,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
)
from mmseg.models.utils import resize


@MODELS.register_module()
class Directioner(EncoderDecoder):
    """
    Directioner, an EncoderDecoder for direction estimation.

    This class overwrites certain methods of EncoderDecoder class to adapt it to direction
    estimation.

    Args:
        backbone (ConfigType): The config for the backbone of directioner.
        decode_head (ConfigType): The config for the decode head of directioner.
        neck (OptConfigType): The config for the neck of directioner. Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of directioner.
            Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The preprocess config of :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model. Defaults to None.
        init_cfg (dict, optional): The weight initialized config for :class:`BaseModule`.
    """

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        """
        Directioner, an EncoderDecoder for direction estimation.

        Args:
            backbone (ConfigType): The config for the backbone of directioner.
            decode_head (ConfigType): The config for the decode head of directioner.
            neck (OptConfigType): The config for the neck of directioner. Defaults to None.
            auxiliary_head (OptConfigType): The config for the auxiliary head of directioner.
            Defaults to None.
            train_cfg (OptConfigType): The config for training. Defaults to None.
            test_cfg (OptConfigType): The config for testing. Defaults to None.
            data_preprocessor (dict, optional): The preprocess config of :class:`BaseDataPreprocessor`.
            pretrained (str, optional): The path for pretrained model. Defaults to None.
            init_cfg (dict, optional): The weight initialized config for :class:`BaseModule`.
        """
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.register_buffer("pi", torch.tensor(np.pi).float())

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.out_channels = 2 * len(self.decode_head.dir_classes)

    def _convert_to_angles(self, vector_field: Tensor) -> Tensor:
        """
        Converts 2D vector field into angular values.

        Args:
            vector_field (Tensor): Per class 2D vector field of shape (K*2, H, W).

        Returns:
            angles (Tensor): Per class direction angle values in radians (from 0 to π) of shape
            (K, H, W).
        """
        h, w = vector_field.shape[-2:]
        vector_field = vector_field.reshape(-1, 2, h, w)  # 2K, H, W -> K, 2, H, W

        x_component: Tensor = vector_field[:, 0, :, :]  # K, H, W
        y_component: Tensor = vector_field[:, 1, :, :]  # K, H, W

        angles: Tensor = torch.atan2(y_component, x_component)  # K, H, W
        angles = (angles + self.pi) / 2

        return angles

    @staticmethod
    def _remove_padding(prediction: Tensor, padding: list[int]) -> Tensor:
        """
        Removes padding from a given tensor based on the specified padding values.

        Args:
            prediction (Tensor): The tensor of shape (C, H, W) to be flipped.
            padding (list[int): A list of integers specifying the padding dimensions in the
            order [left, right, top, bottom].

        Returns:
            Tensor: Tensor without padding.
        """
        padding_left, padding_right, padding_top, padding_bottom = padding
        _, h, w = prediction.shape

        return prediction[:, padding_top: h - padding_bottom, padding_left: w - padding_right]

    @staticmethod
    def _unflip(prediction: Tensor, flip_type: str) -> Tensor:
        """
        Flips a given prediction tensor along a specified axis.

        Args:
            prediction (Tensor): The tensor of shape (C, H, W) to be flipped.
            flip_type (str): A string specifying the flip type.

        Returns:
            Tensor: Flipped vector field tensor.
        """
        if flip_type not in ["horizontal", "vertical"]:
            raise ValueError(f"flip type {flip_type} is not supported.")

        if flip_type == "horizontal":
            return prediction.flip(dims=(2,))

        return prediction.flip(dims=(1,))

    def postprocess_result(
        self, vector_fields: Tensor, data_samples: OptSampleList = None
    ) -> SampleList:
        """
        Converts the list of results to `SegDataSample`.

        Args:
           vector_fields (Tensor): The estimated directions in the form of 2D vector field for
               each input image.
           data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually includes
               information such as `metainfo` and `gt_sem_seg`. Default to None.

        Returns:
           list[:obj:`SegDataSample`]: Direction estimation results of the input images.
               Each SegDataSample usually contains:
               - ``estimated_dirs``(PixelData): Estimated directions as per pixel angle.
               - ``estimated_vs``(PixelData): Estimated directions as per pixel 2D vector, before
                   conversion into angles.
               - ``dir_classes``(list): A list of direction classes.
        """
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(vector_fields)]

        for vector_field, data_sample in zip(vector_fields, data_samples):

            if "img_padding_size" in data_sample.metainfo:
                padding = data_sample.metainfo["img_padding_size"]
                vector_field = self._remove_padding(vector_field, padding)

            if "padding_size" in data_sample.metainfo:
                padding = data_sample.metainfo["padding_size" ]
                vector_field = self._remove_padding(vector_field, padding)

            if "filp" in data_sample.metainfo:
                flip = data_sample.metainfo["flip"]
                vector_field = self._unflip(vector_field, flip)

            if "img_shape" in data_sample.metainfo:
                image_size = data_sample.metainfo["img_shape"]
                vector_field = resize(
                    vector_field,
                    size=image_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                )

            angles = self._convert_to_angles(vector_field)

            data_sample.set_data(
                {
                    "estimated_vs": PixelData(**{"data": vector_field}),
                    "estimated_dirs": PixelData(**{"data": angles}),
                    "dir_classes": self.decode_head.dir_classes,
                }
            )

        return data_samples
