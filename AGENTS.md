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
