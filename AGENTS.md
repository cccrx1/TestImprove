# Repository Guidelines

## Project Structure & Module Organization
`core/` contains reusable library code: attacks in `core/attacks/`, defenses in `core/defenses/`, models in `core/models/`, and shared utilities in `core/utils/`. `workflows/` holds the entry points for experiments such as `train_clean.py`, `run_attack.py`, and `run_refine.py`. `configs/` stores JSON templates for those workflows, `scripts/` contains one-off data preparation helpers, `example_model/` provides sample weights, and `outputs/` is the default location for generated checkpoints and reports.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`, then install a matching `torch`/`torchvision` build for your CUDA environment as noted in `requirements.txt`.

Run the main workflows from the repository root:

- `python -m workflows.train_clean --config configs/train_clean.template.json` trains a clean baseline.
- `python -m workflows.run_attack --config configs/run_attack.template.json` trains and evaluates a backdoored model.
- `python -m workflows.run_refine --config configs/run_refine.template.json` runs REFINE or a configured variant.
- `python scripts/prepare_cub200_dataset.py --archive-path <archive> --output-root <dir>` converts CUB-200 into ImageFolder layout.

Copy a template from `configs/`, rename it for the experiment, and edit only the fields you need.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and concise module-level imports. Keep workflow modules thin and push reusable logic into `core/` or `workflows/common.py`. Use descriptive config filenames such as `run_attack.cifar10.badnets.json`.

## Testing Guidelines
This repository does not currently include a dedicated automated test suite. Validate changes by running the affected workflow with a small config and checking artifacts under `outputs/`. For new functionality, prefer adding a minimal reproducible config and verify both console output and saved checkpoints or reports.

## Commit & Pull Request Guidelines
Recent history uses short commit subjects such as `update`; contributors should raise the bar and write imperative, specific messages like `Add WaNet grid persistence`. Keep commits scoped to one change. Pull requests should summarize the experiment or code path affected, list the config used for validation, link related issues, and include key metrics or screenshots when output reports change.

## Configuration & Output Tips
Do not commit large datasets, generated checkpoints, or populated `outputs/` directories. Keep secrets and machine-specific paths out of committed JSON configs; use copied local configs when paths differ across environments.

## REFINE Research Constraints
Treat REFINE and REFINE_GC as research methods whose core logic should not be changed for routine experiment setup. The core REFINE boundary includes the UNet input transformation, optional output label mapping, frozen backdoored classifier, and the original training objective:

- REFINE: `cls_loss + lmd * supconloss`
- REFINE_GC: `cls_loss + lmd * supconloss + grid_reg_weight * grid_loss`

Do not add new loss terms, rescale loss components, bypass the UNet, or alter prediction/evaluation semantics unless the user explicitly asks to change the method. Prefer changing JSON config values that the implementation already exposes.

For CUB-200 experiments, use the project’s established stable REFINE settings: ImageNet normalization, `unet_kwargs.first_channels: 64`, `batch_size: 16`, `lr: 0.001`, schedule `[5, 8]`, `lmd: 0.05`, and `enable_label_shuffle: false`. Disabling label shuffle does not bypass REFINE; it selects the identity output mapping so the UNet still learns input reprogramming without requiring a 200-class random label permutation.

REFINE_GC is intended as a low-risk geometric enhancement. It should especially help geometric/warping attacks such as WaNet while preserving comparable behavior on non-geometric attacks like BadNets or Blended. When interpreting results, do not claim that GC must improve every attack; the intended claim is stronger robustness for geometric triggers without materially harming other defenses.

When preparing defense configs, the model checkpoint must be the trained backdoored model under `outputs/<dataset>/attacks/.../ckpt_epoch_*.pth`. In attack summary rows, a `checkpoint` column may refer to the clean checkpoint used to initialize attack training; do not use that as the defense target unless it actually points to the attack run directory.
