# REFINE Research Plan

## Current repository assessment

This repository already contains:

- multiple backdoor attacks
- multiple defenses
- model architectures and training utilities
- logging and evaluation helpers

For the thesis direction of backdoor defense based on `REFINE`, the current codebase is usable as a starting point, but the `REFINE` implementation itself needed basic correctness fixes before running systematic experiments.

## Immediate engineering priorities

### 1. Stabilize the REFINE baseline

Finish these before claiming any improvement:

- ensure `REFINE.predict()` and `REFINE.test()` run without undefined variables
- remove device-hardcoded logic such as unconditional `.cuda()`
- infer or pass `num_classes` consistently
- make evaluation robust for datasets with class count smaller than 5
- make `ASR_NoTarget` filtering generic instead of hardcoding only `CIFAR10`, `MNIST`, and `DatasetFolder`

### 2. Build a multi-dataset experiment path

Recommended benchmark order:

1. `CIFAR10`
2. `GTSRB` or `Tiny-ImageNet` via `DatasetFolder`
3. one grayscale dataset if you want to test channel sensitivity, such as `MNIST`

For each dataset, keep the following fixed template:

- benign model architecture
- attack type
- poison rate
- target label
- clean accuracy
- attack success rate
- defended clean accuracy
- defended attack success rate

## REFINE improvement directions

Two main directions are technically reasonable.

### Direction A: improve the transformation module inside REFINE

Candidate modifications:

- replace plain reconstruction-style `UNet` with a lighter or task-aware transformation network
- add a consistency loss between clean prediction and defended prediction
- add a feature-level alignment loss using intermediate model features
- add a perturbation budget penalty such as `L1`, `L2`, or total variation to stop over-transformation
- add confidence-aware weighting so low-confidence samples receive stronger refinement

Recommended first ablation:

`BCELoss + SupConLoss + lambda_rec * ||x_adv - x||_1`

This is the lowest-risk extension because it preserves the current training structure and is easy to analyze in the thesis.

### Direction B: improve sample-adaptive behavior

Candidate modifications:

- predict a per-sample refinement strength
- gate refinement by confidence gap before and after transformation
- run a two-branch version: weak refine for likely clean samples, strong refine for suspicious samples

This direction is more novel, but it has higher implementation and debugging cost.

## Thesis-friendly experiment design

Recommended ablation table:

1. baseline REFINE
2. REFINE + reconstruction regularization
3. REFINE + feature alignment
4. REFINE + adaptive weighting

Recommended attack coverage:

1. patch-based trigger: `BadNets`
2. blending trigger: `Blended`
3. geometric trigger: `WaNet`
4. reflection or physical-style trigger: `Refool` or `PhysicalBA`

Recommended metrics:

- clean accuracy
- ASR
- ASR without target-class source samples
- defended model inference time
- refinement module training time

## Suggested next implementation steps

1. add a configurable regularization term to `REFINE.train_unet()`
2. expose loss weights in a unified schedule/config structure
3. create a reusable experiment script for dataset/model/attack/defense combinations
4. add one `DatasetFolder`-based benchmark path first, then expand to more datasets
5. write result tables directly from logs into a summary file for the thesis

## Practical recommendation

The safest first paper/thesis path is:

1. reproduce baseline REFINE on one dataset
2. add one simple but defensible loss term
3. validate on multiple attacks
4. then extend to multiple datasets

This keeps the work technically coherent and reduces the risk of ending up with a broad but weak story.
