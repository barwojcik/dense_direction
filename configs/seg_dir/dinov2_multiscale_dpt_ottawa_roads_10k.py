_base_ = [
    "../_base_/base_config.py",
    "../_base_/datasets/ottawa_roads.py",
    "../_base_/schedulers/adamw_onecycle_10k.py",
    "../_base_/models/heads/dpt_dual_w_multiscale.py",
]
work_dir = "./outputs/seg_dir/ottawa"
model = dict(
    type="SegmentoDirectioner",
    decode_head=dict(
        dir_head=dict(
            real_head=dict(
                dir_classes=[0],
            ),
        ),
    ),
)
val_evaluator = [
    dict(
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
    ),
    dict(
        type="IoUMetric",
        iou_metrics=[
            "mDice",
        ],
    ),
]
test_evaluator = val_evaluator
