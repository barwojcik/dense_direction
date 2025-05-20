"""
Directioner meta architecture.

This module provides the Directioner class, a modification to EncoderDecoder class that adopts it
to direction estimation.
"""

from typing import Optional

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

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners

    @staticmethod
    def _convert_to_angles(vector_filed: Tensor) -> Tensor:
        """
        Converts 2D vector field into angular values.

        Args:
            vector_filed (Tensor): Per class 2D vector field of shape (K, 2, H, W).

        Returns:
            angles (Tensor): Per class direction angle values in radians (from π/2 to -π/2) of shape
            (K, H, W).
        """
        x_component: Tensor = vector_filed[:, 0, :, :]  # K, H, W
        y_component: Tensor = vector_filed[:, 1, :, :]  # K, H, W

        angles: Tensor = torch.atan2(x_component, y_component)  # K, H, W
        angles = angles / 2

        return angles

    def postprocess_result(
        self, dir_vector_field: Tensor, data_samples: OptSampleList = None
    ) -> SampleList:
        """
        Converts the list of results to `SegDataSample`.

        Args:
           dir_vector_field (Tensor): The estimated directions in the form of 2D vector field for
               each input image.
           data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually includes
               information such as `metainfo` and `gt_sem_seg`. Default to None.

        Returns:
           list[:obj:`SegDataSample`]: Direction estimation results of the input images.
               Each SegDataSample usually contains:

           - ``estimated_dirs``(PixelData): Estimated directions as per pixel angle.
           - ``estimated_vs``(PixelData): Estimated directions as per pixel 2D vector, before
               conversion into angles.
        """
        batch_size, k, _, h1, w1 = dir_vector_field.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if "img_padding_size" not in img_meta:
                    padding_size = img_meta.get("padding_size", [0] * 4)
                else:
                    padding_size = img_meta["img_padding_size"]
                padding_left, padding_right, padding_top, padding_bottom = padding_size
                # i_dir_vfield shape is 1, k, 2, h, w after remove padding
                i_dir_vfield = dir_vector_field[
                    i : i + 1, :, :, padding_top : h1 - padding_bottom, padding_left : w1 - padding_right
                ]

                flip = img_meta.get("flip", None)
                if flip:
                    flip_direction = img_meta.get("flip_direction", None)
                    assert flip_direction in ["horizontal", "vertical"]
                    if flip_direction == "horizontal":
                        i_dir_vfield = i_dir_vfield.flip(dims=(4,))
                    else:
                        i_dir_vfield = i_dir_vfield.flip(dims=(3,))

                # resize as original shape
                h2, w2 = img_meta["ori_shape"]
                i_dir_vfield = resize(
                    i_dir_vfield.view(1, k*2, h1, w1),
                    size=img_meta["ori_shape"],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                ).reshape(k, 2, h2, w2)
            else:
                i_dir_vfield = dir_vector_field[i]

            i_dir_angles = self._convert_to_angles(i_dir_vfield)

            data_samples[i].set_data(
                {
                    "estimated_vs": PixelData(**{"data": i_dir_vfield}),
                    "estimated_dirs": PixelData(**{"data": i_dir_angles}),
                }
            )

        return data_samples
