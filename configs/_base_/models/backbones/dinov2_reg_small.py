model_crop_size = (28 * 14, 28 * 14)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=model_crop_size,
)
model = dict(
    type="Directioner",
    backbone=dict(
        type="Dino2TorchHub",
        model_size="small",
        with_registers=True,
        layers_to_extract=[2, 5, 8, 11],
        return_class_token=True,
        frozen=True,
    ),
    test_cfg=dict(
        mode="slide",
        stride=(26 * 14, 26 * 14),
        crop_size=model_crop_size,
    ),
)
