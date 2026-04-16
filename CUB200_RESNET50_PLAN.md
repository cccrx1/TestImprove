# CUB200 + ResNet50 Experiment Plan

## Goal

Use a stronger clean baseline on `CUB200` to re-evaluate the input-purification defense line.

Current main question:

- On `CUB200 + ResNet50`, how does the original input-purification defense perform under different attacks?
- Which attacks are easier to defend against, and which are harder?
- How much clean accuracy is sacrificed to reduce ASR?

This stage does **not** prioritize designing a new method first.
It prioritizes building a reliable experiment chain:

1. strong clean model
2. attack model
3. original defense
4. result comparison

## Baseline Already Obtained

Current clean result:

- dataset: `CUB200`
- model: `torchvision-resnet50`
- epochs: `30`
- top1: `0.8008`
- top5: `0.9439`

This clean baseline is strong enough to support the next attack/defense experiments.

## Phase 1: Clean Baseline

Config:

- [configs/examples/cub200_clean_torchvision-resnet50.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_clean_torchvision-resnet50.json:1)

Command:

```bash
python -m workflows.train_clean --config configs/examples/cub200_clean_torchvision-resnet50.json
```

Target:

- keep clean `Top-1` around `0.78` to `0.82`

Output to record:

- clean checkpoint path
- `Top-1`
- `Top-5`
- final epoch

## Phase 2: Attack Training

Use the same clean baseline to train three attacks.

### 2.1 BadNets

Config:

- [configs/examples/cub200_badnets_torchvision-resnet50_attack_e30.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_badnets_torchvision-resnet50_attack_e30.json:1)

Command:

```bash
python -m workflows.run_attack --config configs/examples/cub200_badnets_torchvision-resnet50_attack_e30.json
```

### 2.2 Blended

Start from:

- [configs/examples/cub200_blended_resnet50_attack.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_blended_resnet50_attack.json:1)

Recommended adjustment:

- change clean checkpoint to the new `ResNet50` clean model
- if needed, extend epochs from `20` to `30`

### 2.3 WaNet

Start from:

- [configs/examples/cub200_wanet_resnet50_attack.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_wanet_resnet50_attack.json:1)

Recommended adjustment:

- change clean checkpoint to the new `ResNet50` clean model
- if needed, extend epochs from `20` to `30`

For each attack, record:

- attack checkpoint
- clean accuracy before defense
- ASR before defense
- poison rate

## Phase 3: Original Defense Evaluation

Do not modify the defense first.
Evaluate the original input-purification defense on all three attacks.

### 3.1 BadNets + defense

Start from:

- [configs/examples/cub200_badnets_resnet50_refine.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_badnets_resnet50_refine.json:1)

Need to update:

- `model.checkpoint`
- use the newly trained `BadNets` attack checkpoint

### 3.2 Blended + defense

Config:

- [configs/examples/cub200_blended_resnet50_refine.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_blended_resnet50_refine.json:1)

Need to update:

- `model.checkpoint`

### 3.3 WaNet + defense

Config:

- [configs/examples/cub200_wanet_resnet50_refine.json](/abs/path/c:/Users/17672/Documents/Projects/TestImprove/configs/examples/cub200_wanet_resnet50_refine.json:1)

Need to update:

- `model.checkpoint`

For each defense experiment, record:

- defense checkpoint
- clean accuracy after defense
- ASR after defense
- clean accuracy drop relative to attacked model

## Phase 4: Comparison Focus

At this stage, do not rush to tune new loss terms.
First answer these questions:

1. Does the original defense still work on `CUB200 + ResNet50`?
2. Is its effect similar across `BadNets`, `Blended`, and `WaNet`?
3. Is the main problem:
   clean accuracy drop, weak ASR reduction, or both?

Recommended comparison table:

| Attack | Clean before defense | ASR before defense | Clean after defense | ASR after defense | Main observation |
| --- | --- | --- | --- | --- | --- |
| BadNets |  |  |  |  |  |
| Blended |  |  |  |  |  |
| WaNet |  |  |  |  |  |

## How This Supports the Thesis

This experiment line supports the thesis in a clearer way than directly forcing a new method.

Main thesis logic:

1. Build a stronger clean baseline on `CUB200`
2. Re-evaluate the original input-purification defense under multiple attacks
3. Compare performance differences across attacks
4. Analyze where the method works and where it struggles
5. Only then decide whether a targeted improvement is necessary

## Immediate Next Step

The next practical order is:

1. finish the `CUB200 + ResNet50` clean baseline
2. train `BadNets` attack with the new clean checkpoint
3. evaluate the original defense on that attack first

Do not start all three attacks in parallel before the first one is confirmed to run correctly.
