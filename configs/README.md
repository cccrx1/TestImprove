This folder stores JSON configs for workflow scripts.

Mapping:

- `train_clean.template.json` -> `python -m workflows.train_clean --config <path>`
- `run_attack.template.json` -> `python -m workflows.run_attack --config <path>`
- `run_refine.template.json` -> `python -m workflows.run_refine --config <path>`

Suggested usage:

1. Copy a template file.
2. Rename it for one concrete experiment.
3. Modify only the fields you need.
4. Run the matching workflow script with `--config`.

Notes:

- Attack selection is name-based, for example `badnets`, `blended`, or `wanet`.
- Poisoning rate is configured inside `attack.kwargs.poisoned_rate`.
- `gtsrb` and `cub200` are supported through `ImageFolder` style train/test directories.
- For folder datasets, you can either provide `train_dir` and `test_dir`, or provide `root` plus `train_subdir` and `test_subdir`.
- You can set `dataset.image_size` to force a resize before `ToTensor()`.
- For high-resolution datasets such as `cub200`, prefer `torchvision-resnet18` or `torchvision-resnet50`.
- You can pass torchvision pretrained weights with `model.kwargs.weights`, for example `"DEFAULT"`.
