_base_ = [
    "../backbones/dinov2_reg_small.py",
]
model = dict(
    neck=dict(
        type="DPTDecoderBlock",
        return_list=True,
        embed_dims=384,
        patch_size=14,
        channels=384,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ),
)
