# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a backdoor attack and defense research framework focused on neural network security. The codebase implements backdoor attacks (BadNets, Blended, WaNet) and defenses (REFINE and variants) for image classification models.

## Architecture

The codebase follows a modular structure:

- `core/attacks/`: Backdoor attack implementations inheriting from `Base` class
  - BadNets: Patch-based trigger injection
  - Blended: Blended trigger overlay
  - WaNet: Warping-based imperceptible triggers
  
- `core/defenses/`: Defense mechanisms inheriting from `Base` class
  - REFINE: Main defense using UNet-based input transformation
  - REFINE_GC: REFINE with geometric correction

- `core/models/`: Neural network architectures
  - ResNet, VGG variants for CIFAR/MNIST
  - UNet for input transformation
  - Autoencoder, geometric correction modules

- `workflows/`: High-level experiment orchestration
  - `run_attack.py`: Train backdoored models
  - `run_refine.py`: Apply REFINE defense to backdoored models
  - `train_clean.py`: Train clean baseline models
  - `common.py`: Shared utilities for dataset loading, model building, attack instantiation
  - `refine_pipeline.py`: Defense pipeline configuration and model building
  - `reporting.py`: Experiment tracking and result logging

- `configs/`: JSON configuration files for experiments
  - Templates define dataset, model, attack/defense parameters, and training schedules
  - Examples in `configs/examples/` show concrete experiment configurations

## Common Commands

### Setup
```bash
# Install dependencies (PyTorch must be installed separately for your CUDA version)
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### Dataset Preparation
```bash
# Prepare CUB-200-2011 dataset (converts to ImageFolder format)
python scripts/prepare_cub200_dataset.py --archive-path datasets/CUB_200_2011.tgz --output-root datasets/cub200
```

### Training Workflows

Train a clean baseline model:
```bash
python -m workflows.train_clean --config configs/train_clean.template.json
```

Run a backdoor attack:
```bash
python -m workflows.run_attack --config configs/run_attack.template.json
```

Apply REFINE defense:
```bash
python -m workflows.run_refine --config configs/run_refine.template.json
```

## Configuration System

All workflows are driven by JSON config files with these key sections:

- `dataset`: Dataset name (cifar10, mnist, gtsrb, cub200), root path, image_size
- `model`: Architecture name (resnet18, resnet50, vgg16), kwargs, checkpoint path
- `attack`: Attack name (badnets, blended, wanet), poisoned_rate, trigger parameters
- `refine`: Defense variant (refine, refine_gc), UNet config, hyperparameters
- `schedule`: Training hyperparameters (epochs, lr, batch_size, device, save_dir)

### Key Configuration Notes

- For high-resolution datasets like CUB-200, use `torchvision-resnet50` with ImageNet pretrained weights (`model.kwargs.weights: "DEFAULT"`)
- Attack triggers are auto-generated if `pattern` and `weight` are null (BadNets/Blended) or if `identity_grid`/`noise_grid` are omitted (WaNet)
- WaNet grids are saved during attack training and auto-loaded during defense if checkpoint path is provided
- Defense variants are selected via `refine.defense_name`: `refine`, `refine_gc`
- Folder-based datasets (GTSRB, CUB-200) require either `train_dir`/`test_dir` or `root` + `train_subdir`/`test_subdir`

## Output Structure

Experiments save to `outputs/` with this hierarchy:
```
outputs/
├── attacks/{dataset}/{attack}_{model}_{attack}_{timestamp}/
│   ├── ckpt_epoch_*.pth
│   ├── identity_grid.pth (WaNet only)
│   └── noise_grid.pth (WaNet only)
├── defenses/{dataset}/{defense}_{model}_{attack}_{timestamp}/
│   └── ckpt_epoch_*.pth
└── clean/{dataset}/clean_{model}_clean_{timestamp}/
    └── ckpt_epoch_*.pth
```

## Development Workflow

1. Copy a template config from `configs/` and customize for your experiment
2. Train a clean baseline or use an existing checkpoint
3. Run attack workflow to create backdoored model
4. Run defense workflow using the backdoored checkpoint
5. Results are logged to console and saved in run directories

## Dataset Support

- Built-in: CIFAR-10, MNIST
- ImageFolder-style: GTSRB, CUB-200-2011 (requires preparation script)
- Custom datasets: Add to `workflows/common.py` dataset loading logic

## Model Support

- CIFAR-style: resnet18, vgg16 (32x32 input)
- ImageNet-style: torchvision-resnet50 (224x224 input, pretrained available)
- Custom models: Register in `core/models/` and update `workflows/refine_pipeline.py`

## Key Implementation Details

- All attacks and defenses inherit from `core.attacks.base.Base` or `core.defenses.base.Base`
- Seed management is handled at workflow level via `set_global_seed()`
- Metrics are computed via `core.utils.accuracy` with configurable top-k
- Attack poisoning happens via custom transform classes (AddTrigger) inserted into dataset pipelines
- REFINE defense uses a UNet to transform inputs before feeding to the backdoored classifier
- Label shuffling in REFINE is controlled by `enable_label_shuffle` parameter
