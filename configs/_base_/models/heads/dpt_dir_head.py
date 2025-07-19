_base_ = [
    "../backbones/dinov2_reg_small.py",
]
model = dict(
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
)