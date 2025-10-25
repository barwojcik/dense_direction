"""
PackCustomInputs transform for packing custom inputs.

This module provides a PackCustomInputs class that packs custom inputs along semantic segmentation
ground truths.
"""

import warnings
from typing import Iterable, Any

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackCustomInputs(BaseTransform):
    """
    Packs the input data for the semantic segmentation and any other custom data.

    The ``img_meta`` item is always populated. The content of the ``img_meta`` dictionary
    depends on ``meta_keys``. By default, this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple (h, w, c). Note that
            images may be zero padded on the bottom/right if the batch tensor is larger than this
            shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        additional_data_keys (Iterable): Additional data keys to be packed.
        meta_keys (Sequence[str], optional): Meta keys to be packed from ``SegDataSample`` and
            collected in ``data[img_metas]``. Default: ``('img_path', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction')``
    """

    DEFAULT_META_KEYS: tuple[str,...] = (
        'img_path',
        'seg_map_path',
        'ori_shape',
        'img_shape',
        'pad_shape',
        'scale_factor',
        'flip',
        'flip_direction',
        'reduce_zero_label',
    )

    def __init__(
        self,
        additional_data_keys: Iterable[str] = ("gt_directions",),
        meta_keys=None,
    ) -> None:
        """
        Initializes the PackCustomInputs class.

        This class is used to pack the input data for the semantic segmentation and any other
        custom data.

        Args:
            additional_data_keys (Iterable): Additional data keys to be packed.
            meta_keys (Sequence[str], optional): Meta keys to be packed from ``SegDataSample`` and
                collected in ``data[img_metas]``. Default: ``('img_path', 'ori_shape', 'img_shape',
                'pad_shape', 'scale_factor', 'flip', 'flip_direction')`
        """
        if meta_keys is not None:
            meta_keys = set(meta_keys)

        self.data_keys: set[str] = set(additional_data_keys)
        self.meta_keys: set[str] = meta_keys or set(self.DEFAULT_META_KEYS)

    def transform(self, results: dict) -> dict:
        """
        Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the sample, containing
                ground truth segmentation map and additional data from `data_keys`.
        """
        packed_results: dict[Any, Any] = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None, ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        for data_key in self.data_keys:
            if data_key not in results:
                warnings.warn(f"Missing data key {data_key} in results {results['sample_idx']}.")
                continue

            data = dict(data=to_tensor(results[data_key]))
            data_sample.set_data({data_key:PixelData(**data)})

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
