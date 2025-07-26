work_dir = "./outputs"
default_scope = "mmseg"
env_cfg = dict(dist_cfg=dict(backend="nccl"))
log_level = "INFO"
log_processor = dict(
    window_size=10,
    by_epoch=False,
    custom_cfg=None,
    num_digits=4,
)
runner = dict(
    type="Runner",
)
