This folder stores JSON configs for workflow scripts.

Mapping:

- `train_clean.template.json` -> `python -m workflows.train_clean --config <path>`
- `run_attack.template.json` -> `python -m workflows.run_attack --config <path>`
- `run_refine.template.json` -> `python -m workflows.run_refine --config <path>`
- For defense variants, set `refine.defense_name`, for example `refine` or `refine_gc`.
- The repository `requirements.txt` follows the lightweight dependency style used in `BiShe`; install `torch` and `torchvision` separately for your server CUDA environment.
- Outputs are grouped as `outputs/<dataset>/<clean|attacks|defenses>/<run_name>/`, with a compact summary table at `outputs/reports/experiment_summary.csv`.

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
- `poisoned_transform_train_index` and `poisoned_transform_test_index` can be left as `null`; workflows will insert the trigger right before `ToTensor()`.
- For `badnets` and `blended`, `pattern` and `weight` can be left as `null`; workflows will generate a default bottom-right square trigger.
- For `wanet`, `identity_grid` and `noise_grid` can be omitted; workflows will generate them from the current image size.
- You can set `dataset.image_size` to force a resize before `ToTensor()`.
- For high-resolution datasets such as `cub200`, prefer `torchvision-resnet50` or another ImageNet-pretrained backbone over CIFAR-style backbones.
- You can pass torchvision pretrained weights with `model.kwargs.weights`, for example `"DEFAULT"`.

REFINE notes:

- Defense `model.checkpoint` should point to the trained backdoored model in `outputs/<dataset>/attacks/.../ckpt_epoch_*.pth`.
- For CUB-200 REFINE/REFINE_GC, prefer the stable settings used in prior runs: ImageNet normalization, `first_channels: 64`, `batch_size: 16`, `lr: 0.001`, schedule `[5, 8]`, `lmd: 0.05`, and `enable_label_shuffle: false`.
- `enable_label_shuffle: false` keeps the output mapping as identity. The UNet is still trained and used; this only avoids a random 200-class output permutation on CUB-200.
- REFINE_GC is primarily expected to help geometric attacks such as WaNet and should not be expected to improve every non-geometric attack.
