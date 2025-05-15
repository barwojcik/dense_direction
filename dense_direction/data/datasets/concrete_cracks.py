"""
ConcreteCracks dataset for semantic segmentation.

This module provides a class for concrete cracks segmentation dataset, which is a collection of images
and corresponding semantic segmentation masks of cracks in concrete surfaces.

Dataset source:
    Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”, Mendeley Data, V1, doi: 10.17632/jwsn7tfbrp.1

"""

from mmengine import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ConcreteCracksDataset(BaseSegDataset):
    """
    Concrete cracks segmentation dataset.

    This class implements a concrete cracks segmentation dataset for semantic segmentation.
    Dataset source:
        Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”, Mendeley Data, V1, doi: 10.17632/jwsn7tfbrp.1

    Arguments:
        data_root (str): Root directory of the dataset.
            Default to './data/concreteCrackSegmentationDataset',
        img_suffix (str): Suffix of the image files.
            Default to '.jpg'.
        seg_mask_suffix (str): Suffix of the segmentation mask files.
            Default to '.jpg'.
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
        data_prefix (str): Path to the folder where the images are stored.
            Default to dict(img_path="rgb", seg_map_path="BW").
        **kwargs: Any other argument that is available in base class.
    """

    METAINFO = dict(
        classes=("background", "crack"),
        palette=[[65, 65, 129], [128, 64, 128]],
    )

    def __init__(
        self,
        data_root: str = "./data/concreteCrackSegmentationDataset",
        img_suffix: str = ".jpg",
        seg_map_suffix: str = ".jpg",
        reduce_zero_label: bool = False,
        data_prefix: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix=data_prefix or dict(img_path="rgb", seg_map_path="BW"),
            **kwargs,
        )
