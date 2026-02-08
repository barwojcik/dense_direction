# Configuration

Dense Direction uses the modular `mmengine` configuration system. Each experiment is defined by a
Python config file that declares the model, datasets, training loop, hooks, and evaluation settings.

## Where configs live
- `configs/_base_`: shared fragments that can be imported into other configs.
- `configs/dir`: direction-only training configs.
- `configs/seg_dir`: joint segmentation + direction configs.

## Common workflow
1) Pick a base config closest to your dataset or model.
2) Override paths and dataset settings for your local environment.
3) Run with `run.py` and optional `--cfg-options` overrides.

For details on the config syntax and available keys, refer to the `mmengine` and `mmseg` docs.
