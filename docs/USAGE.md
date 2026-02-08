# Usage

## Train/test entry point
All experiments run through `run.py`, which wraps an `mmengine` runner built from a config file.

### Common commands
Train and test with a single config:
```bash
python run.py --config configs/dir/dinov2_dpt_concrete_cracks_10k.py
```

Train only:
```bash
python run.py --phase train --config configs/dir/dinov2_dpt_concrete_cracks_10k.py
```

Test only:
```bash
python run.py --phase test --config configs/dir/dinov2_dpt_concrete_cracks_10k.py
```

Override config values inline:
```bash
python run.py --config configs/dir/dinov2_dpt_concrete_cracks_10k.py \
  --cfg-options runner.max_epochs=5
```

## Tips
- Use `configs/dir` for direction-only training.
- Use `configs/seg_dir` for joint segmentation + direction variants.
