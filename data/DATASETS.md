# Datasets

This file lists datasets available in `dense_direction` library.

## Concrete Cracks Segmentation Dataset

### Description

> The dataset includes 458 hi-res images together with their alpha maps (BW) indicating the crack presence. The ground truth for semantic segmentation has two classes to conduct binary pixelwise classification. The photos are captured in various buildings located in Middle East Technical University. 
>
> -- <cite> Özgenel, Çağlar Fırat (2019), “Concrete Crack Segmentation Dataset”, Mendeley Data, V1, doi: 10.17632/jwsn7tfbrp.1, url: https://data.mendeley.com/datasets/jwsn7tfbrp/1</cite>

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
