"""
ConcreteCracks dataset for semantic segmentation.

This module provides a class for concrete cracks segmentation dataset, which is a collection of
images and corresponding semantic segmentation masks of cracks in concrete surfaces.

Dataset source:
    Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”,
    Mendeley Data, V1, doi: https://doi.org/10.17632/jwsn7tfbrp.1
"""

import copy
from typing import Any
from mmengine import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ConcreteCracksDataset(BaseSegDataset):
    """
    Concrete cracks segmentation dataset.

    This class implements a concrete cracks segmentation dataset for semantic segmentation.
    For more details, refer to the .data/DATASETS.md.

    Dataset source:
        Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”,
        Mendeley Data, V1, doi: https://doi.org/10.17632/jwsn7tfbrp.1

    Arguments:
        data_root (str): Root directory of the dataset.
            Default to './data/concreteCrackSegmentationDataset',
        phase (str): 'train', 'val', 'test' or None (default: None)
        **kwargs: Any other argument that is available in base class except defaults ('img_suffix',
            'seg_map_suffix', 'reduce_zero_label', 'data_prefix', 'indices').
    """

    METAINFO: dict[str, Any] = dict(
        classes=("background", "crack"),
        palette=[[65, 65, 129], [128, 64, 128]],
    )
    DEFAULT_PARAMS: dict[str, Any] = dict(
        img_suffix=".jpg",
        seg_map_suffix=".jpg",
        reduce_zero_label=False,
        data_prefix=dict(img_path="rgb", seg_map_path="BW"),
    )
    SPLITS: dict[str, int] = dict(
        train=278,
        val=[i for i in range(278, 368)],
        test=[i for i in range(368, 458)],
    )

    def __init__(
        self,
        data_root: str = "./data/concreteCrackSegmentationDataset",
        phase: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the ConcreteCracksDataset class.

        Arguments:
            data_root (str): Root directory of the dataset.
                Default to './data/concreteCrackSegmentationDataset',
            phase (str): 'train', 'val', 'test' or None (default: None)
            **kwargs: Any other argument that is available in base class except defaults
                ('img_suffix', 'seg_map_suffix', 'reduce_zero_label', 'data_prefix', 'indices').
        """
        parameters = copy.deepcopy(self.DEFAULT_PARAMS)

        if phase is not None and phase in self.SPLITS.keys():
            parameters["indices"] = self.SPLITS[phase]

        kwargs.update(parameters)
        super().__init__(
            data_root=data_root,
            **kwargs,
        )
