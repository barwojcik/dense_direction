"""
SegmentoDirectioner meta architecture.

This module provides the SegmentoDirectioner class, a modification to EncoderDecoder class that
adopts it to perform both segmentation and direction estimation tasks simultaneously.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (
    ConfigType,
    OptSampleList,
    SampleList,
)

from .directioner import Directioner


@MODELS.register_module()
class SegmentoDirectioner(Directioner):
    """
    SegmentoDirectioner, an EncoderDecoder for simultaneous semantic segmentation and direction
    estimation.

    This class overwrites certain methods of EncoderDecoder class to adapt it to perform both
    segmentation and direction estimation tasks simultaneously.

    Args:
        backbone (ConfigType): The config for the backbone.
        decode_head (ConfigType): The config for the decode head.
        neck (OptConfigType): The config for the neck. Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head.
            Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The preprocess config of :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model. Defaults to None.
        init_cfg (dict, optional): The weight initialized config for :class:`BaseModule`.
    """

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels
        self.out_dir_channels = 2 * len(self.decode_head.dir_classes)

    def _get_seg_pred(self, seg_logit: Tensor) -> Tensor:
        """

        """
        if seg_logit.shape[0] == 1:
            seg_logit = seg_logit.sigmoid()
            return (seg_logit > self.decode_head.threshold).to(seg_logit)

        return seg_logit.argmax(dim=0, keepdim=True)

    def postprocess_seg_result(
        self, seg_logits: Tensor, data_samples: OptSampleList = None
    ) -> SampleList:
        """Convert segmentation results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from a model of each input
                image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually includes
                information such as `metainfo` and `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the input images.
                Each SegDataSample usually contain:
                - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
                - ``seg_logits``(PixelData): Predicted logits of semantic segmentation before
                    normalization.
                - ``estimated_dirs``(PixelData): Estimated directions as per pixel angle.
                - ``estimated_vs``(PixelData): Estimated directions as per pixel 2D vector, before
                   conversion into angles.
                - ``dir_classes``(list): A list of direction classes.
        """
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(seg_logits)]

        for seg_logit, data_sample in zip(seg_logits, data_samples):
            seg_logit = self._transform_prediction(seg_logit, data_sample)
            seg_pred = self._get_seg_pred(seg_logit)

            data_sample.set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_logit}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred}),
                }
            )

        return data_samples

    def postprocess_result(
        self, results: tuple[Tensor, Tensor], data_samples: OptSampleList = None
    ) -> SampleList:
        """
        Converts the list of results to `SegDataSample`.

        Args:
           results (tuple[Tensor, Tensor]): Tuple containing segmentation and direction estimation
               results from forward pass.
           data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually includes
               information such as `metainfo` and `gt_sem_seg`. Default to None.

        Returns:
           list[:obj:`SegDataSample`]: Direction estimation results of the input images.
               Each SegDataSample usually contains:

           - ``estimated_dirs``(PixelData): Estimated directions as per pixel angle.
           - ``estimated_vs``(PixelData): Estimated directions as per pixel 2D vector, before
               conversion into angles.
        """
        seg_logits, dir_vector_field = results

        # it's equivalent to EncoderDecoder.postprocess_result()
        data_samples = self.postprocess_seg_result(seg_logits, data_samples)

        # call to Directioner postprocess_result method
        data_samples = super().postprocess_result(dir_vector_field, data_samples)

        return data_samples

    def slide_inference(self, inputs: Tensor, batch_img_metas: list[dict]) -> tuple[Tensor, Tensor]:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            tuple[Tensor, Tensor]: The segmentation and direction results for each input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_seg_channels = self.out_channels
        out_dir_channels = self.out_dir_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        seg_preds = inputs.new_zeros((batch_size, out_seg_channels, h_img, w_img))
        dir_preds = inputs.new_zeros((batch_size, out_dir_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]["img_shape"] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit, crop_dir_vectors = self.encode_decode(crop_img, batch_img_metas)
                seg_preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(seg_preds.shape[3] - x2), int(y1), int(seg_preds.shape[2] - y2)),
                )
                dir_preds += F.pad(
                    crop_dir_vectors,
                    (int(x1), int(dir_preds.shape[3] - x2), int(y1), int(dir_preds.shape[2] - y2)),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = seg_preds / count_mat
        dir_vector_field = dir_preds / count_mat

        return seg_logits, dir_vector_field

    def aug_test(
        self, inputs: Tensor, batch_img_metas: list[dict], rescale=True
    ) -> tuple[Tensor, Tensor]:
        """Test with augmentations.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which contains all images in
                the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may also contain:
                'img_shape', 'scale_factor', 'flip', 'img_path', 'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            rescale (bool, optional): If True, resize inputs before augmentation.

        Returns:
            tuple[Tensor, Tensor]: The tuple containing results, seg_logits and dir_vector_field,
                of each input image.
        """
        raise NotImplementedError
