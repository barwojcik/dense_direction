train_batch_size = 8
train_num_workers = 10
test_batch_size = 1
test_num_workers = 3
crop_size = (28 * 14, 28 * 14)
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
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
val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=test_num_workers,
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
test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=test_num_workers,
    collate_fn=dict(type="default_collate"),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="ConcreteCracksDataset",
        phase="test",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="BinarizeAnnotations"),
            dict(type="PackSegInputs"),
        ],
    ),
)
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
test_evaluator = val_evaluator
