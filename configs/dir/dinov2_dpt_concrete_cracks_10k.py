_base_ = [
    "../base_config.py",
]
work_dir = "./outputs/dir/cracks"
train_iters = 10_000
train_batch_size = 8
crop_size = (28 * 14, 28 * 14)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.01,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
)
param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=0.001,
        total_steps=train_iters,
        by_epoch=False,
    )
]
train_cfg = dict(
    by_epoch=False,
    max_iters=train_iters,
    val_interval=1_000,
)
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    collate_fn=dict(type="default_collate"),
    sampler=dict(
        type="InfiniteSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="ConcreteCracksDataset",
        phase="train",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="BinarizeAnnotations"),
            dict(
                type="MaskGuidedRandomCrop",
                crop_size=crop_size,
                min_ratio=0.05,
                max_ratio=0.2,
            ),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(type="PackSegInputs"),
        ],
    ),
)
val_cfg = dict(type="ValLoop")
val_evaluator = dict(
    type="DirectionalLossMetric",
    loss_config=dict(
        loss_name="loss_dir",
        type="EfficientDirectionalLoss",
        reduction="none",
        div=180,
        pad=10,
        squish_values=True,
        norm_values=True,
        norm_order=1,
        mask_patches=True,
        patch_thr=0.98,
        loss_weight=1,
        kernel_cfg=dict(
            type="radial_line_kernel",
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    collate_fn=dict(type="default_collate"),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="ConcreteCracksDataset",
        phase="val",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="BinarizeAnnotations"),
            dict(type="PackSegInputs"),
        ],
    ),
)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)
model = dict(
    type="Directioner",
    backbone=dict(
        type="Dino2TorchHub",
        layers_to_extract=[2, 5, 8, 11],
        return_class_token=True,
        frozen=True,
    ),
    decode_head=dict(
        type="DPTDirectionHead",
        in_index=[0, 1, 2, 3],
        embed_dims=384,
        patch_size=14,
        in_channels=[384, 384, 384, 384],
        input_transform="multiple_select",
        channels=384,
        dropout_ratio=0,
        gt_scale_factor=1.0,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                loss_name="loss_dir",
                type="EfficientDirectionalLoss",
                div=180,
                pad=10,
                squish_values=True,
                norm_values=True,
                norm_order=1,
                mask_patches=True,
                patch_thr=0.98,
                loss_weight=1,
                kernel_cfg=dict(
                    type="radial_line_kernel",
                ),
            ),
        ],
    ),
    test_cfg=dict(
        mode="slide",
        stride=(26 * 14, 26 * 14),
        crop_size=crop_size,
    ),
)