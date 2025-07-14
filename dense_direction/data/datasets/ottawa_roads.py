"""
OttawaRoads dataset for semantic segmentation.

This module provides a class for Ottawa roads segmentation dataset, which is a collection of
satellite images and corresponding semantic segmentation masks of roads in the city of Ottawa,
Canada.

Dataset source:
    Y. Liu, J. Yao, X. Lu, M. Xia, X. Wang and Y. Liu, "RoadNet: Learning to Comprehensively
    Analyze Road Networks in Complex Urban Scenes From High-Resolution Remotely Sensed Images," in
    IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 4, pp. 2043-2056, April 2019,
    doi: https://doi.org/10.1109/TGRS.2018.2870871, github: https://github.com/yhlleo/RoadNet
"""

import copy
from typing import Any
from mmengine import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OttawaRoadsDataset(BaseSegDataset):
    """
    OttawaRoads dataset for semantic segmentation.

    This class implements a road segmentation dataset for semantic segmentation.
    For more details, refer to the .data/DATASETS.md.

    Dataset source:
        Y. Liu, J. Yao, X. Lu, M. Xia, X. Wang and Y. Liu, "RoadNet: Learning to Comprehensively
        Analyze Road Networks in Complex Urban Scenes From High-Resolution Remotely Sensed Images,"
        in IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 4, pp. 2043-2056,
        April 2019, doi: https://doi.org/10.1109/TGRS.2018.2870871,
        github: https://github.com/yhlleo/RoadNet

    Arguments:
        data_root (str): Root directory of the dataset.
            Default to './data/Ottawa-Dataset',
        phase (str): 'train', 'val', 'test' or None (default: None)
        **kwargs: Any other argument that is available in base class except defaults ('img_suffix',
            'seg_map_suffix', 'reduce_zero_label', 'data_prefix', 'indices').
    """

    METAINFO: dict[str, Any] = dict(
        classes=("background", "road"),
        palette=[[65, 65, 129], [128, 64, 128]],
    )
    DEFAULT_PARAMS: dict[str, Any] = dict(
        img_suffix=".tif",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        data_prefix=dict(img_path="images", seg_map_path="masks"),
    )
    SPLITS: dict[str, int] = dict(
        train=12,
        val=[i for i in range(12, 16)],
        test=[i for i in range(16, 20)],
    )

    def __init__(
        self,
        data_root: str = "./data/Ottawa-Dataset",
        phase: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the OttawaRoadsDataset class.

        Arguments:
            data_root (str): Root directory of the dataset.
                Default to './data/Ottawa-Dataset',
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
