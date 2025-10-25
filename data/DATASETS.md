# Datasets

This file lists custom datasets implemented in `dense_direction`. Apart from these, all datasets available in `mmseg` are supported.

More information can be found in `mmseg` documentation about 
[datasets](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/advanced_guides/datasets.md), 
and [adding a new dataset](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/advanced_guides/add_datasets.md).

## Concrete Cracks Segmentation Dataset

### Description

> The dataset includes 458 hi-res images together with their alpha maps (BW) indicating the crack presence. The ground truth for semantic segmentation has two classes to conduct binary pixelwise classification. The photos are captured in various buildings located in Middle East Technical University. 
>
> <cite> Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”, Mendeley Data, V1, doi: https://doi.org/10.17632/jwsn7tfbrp.1, url: https://data.mendeley.com/datasets/jwsn7tfbrp/1</cite>

### Instructions

Download dataset and unpack it in the `data` directory.

Currently, there's a mix of images with `.jpg` and `.JPG` suffixes in `rgb` directory, which can cause issues with
`BaseSegDataset`. To resolve this, you can use the following command to lowercase all image suffixes:

```bash
rename 's/\.JPG$/.jpg/' *.JPG
```

### Dataset structure
```
└── data
    └── concreteCrackSegmentationDataset
        ├── BW
        │   ├── 001.jpg
        │   ├── 011.jpg
        │   ├── ...
        │   └── 610.jpg
        └── rgb
            ├── 001.jpg
            ├── 011.jpg
            ├── ...
            └── 610.JPG
```

## Ottawa Roads Segmentation Dataset

### Description

The dataset contains 20 satellite images of the city of Ottawa, Canada, with annotations for segmentation of roads, their centerline and edges.

Dataset source:

<cite> Y. Liu, J. Yao, X. Lu, M. Xia, X. Wang and Y. Liu, "RoadNet: Learning to Comprehensively
Analyze Road Networks in Complex Urban Scenes From High-Resolution Remotely Sensed Images," in
IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 4, pp. 2043-2056, April 2019,
doi: https://doi.org/10.1109/TGRS.2018.2870871, GitHub: https://github.com/yhlleo/RoadNet </cite>

### Instructions

Download dataset and move `.zip` archive into the `data` directory.

Original dataset structure is not compatible with `BaseSegDataset`, to unpack and rearrange data use the script below:

```bash
python ../tools/datasets/ottawa_roads.py --zip_path ./Ottawa-Dataset.zip --output_dir ./
```

### Dataset structure
```
└── data
    └── Ottawa-Dataset
        ├── centerline
        │   ├── Ottawa-1.png
        │   ├── Ottawa-2.png
        │   ├── ...
        │   └── Ottawa-20.png
        ├── images
        │   ├── Ottawa-1.tif
        │   ├── Ottawa-2.tif
        │   ├── ...
        │   └── Ottawa-20.tif
        └── masks
            ├── Ottawa-1.png
            ├── Ottawa-2.png
            ├── ...
            └── Ottawa-20.png
```