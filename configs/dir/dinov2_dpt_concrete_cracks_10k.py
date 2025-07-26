_base_ = [
    "../_base_/base_config.py",
    "../_base_/datasets/concrete_cracks.py",
    "../_base_/schedulers/adamw_onecycle_10k.py",
    "../_base_/models/heads/dpt_dir_head.py",
]
work_dir = "./outputs/dir/cracks"
