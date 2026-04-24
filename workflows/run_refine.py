import argparse
from copy import deepcopy
import os.path as osp

import torch

from core.utils import resolve_output_dir

from workflows.common import (
    build_attack,
    build_loss,
    complete_schedule,
    infer_in_channels,
    infer_num_classes,
    load_datasets,
    load_json,
    set_global_seed,
)
from workflows.refine_pipeline import RefinePipelineConfig, build_model, build_refine_defense
from workflows.reporting import (
    append_experiment_record,
    build_refine_record,
)


def main():
    parser = argparse.ArgumentParser(description='Run REFINE defense workflow.')
    parser.add_argument('--config', required=True, help='Path to a JSON config file.')
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_global_seed(cfg.get('seed', 0))

    if cfg.get('attack', {}).get('name', '').lower() == 'wanet':
        attack_kwargs = cfg.setdefault('attack', {}).setdefault('kwargs', {})
        if (
            attack_kwargs.get('identity_grid') is None
            and attack_kwargs.get('noise_grid') is None
            and cfg.get('model', {}).get('checkpoint')
        ):
            attack_run_dir = osp.dirname(cfg['model']['checkpoint'])
            identity_grid_path = osp.join(attack_run_dir, 'identity_grid.pth')
            noise_grid_path = osp.join(attack_run_dir, 'noise_grid.pth')
            if osp.isfile(identity_grid_path) and osp.isfile(noise_grid_path):
                attack_kwargs['identity_grid'] = torch.load(identity_grid_path, map_location='cpu')
                attack_kwargs['noise_grid'] = torch.load(noise_grid_path, map_location='cpu')

    train_dataset, test_dataset = load_datasets(cfg['dataset'])
    num_classes = cfg.get('num_classes', infer_num_classes(train_dataset))
    in_channels = cfg.get('in_channels', infer_in_channels(train_dataset))

    pipeline_cfg = RefinePipelineConfig(
        num_classes=num_classes,
        in_channels=in_channels,
        model_name=cfg['model']['name'],
        model_kwargs=cfg['model'].get('kwargs', {}),
        unet_name=cfg['refine'].get('unet_name', 'unet-little'),
        unet_kwargs=cfg['refine'].get('unet_kwargs', {'first_channels': 16}),
        defense_name=cfg['refine'].get('defense_name', 'refine'),
        refine_kwargs=cfg['refine'].get('defense_kwargs', {}),
    )

    model = build_model(
        pipeline_cfg.model_name,
        num_classes=pipeline_cfg.num_classes,
        **pipeline_cfg.model_kwargs,
    )
    if cfg['model'].get('checkpoint'):
        model.load_state_dict(torch.load(cfg['model']['checkpoint'], map_location='cpu'), strict=False)

    refine = build_refine_defense(pipeline_cfg, model=model)

    schedule = complete_schedule(
        cfg['schedule'],
        dataset_name=cfg['dataset']['name'],
        model_name=cfg['model']['name'],
        attack_name=cfg['attack']['name'],
        defense_name=cfg['refine'].get('defense_name', 'refine'),
    )
    method_name = cfg['refine'].get('defense_name', 'refine')
    train_run_dir = None
    clean_run_dir = None
    asr_run_dir = None
    shared_run_dir = None

    if cfg.get('train_unet', True):
        shared_run_dir = refine.train_unet(train_dataset, test_dataset, schedule)
        train_run_dir = shared_run_dir
    else:
        shared_run_dir = resolve_output_dir(schedule, stage='defenses', method_name=method_name)
        train_run_dir = shared_run_dir

    schedule['run_dir'] = shared_run_dir

    if cfg.get('eval_clean', True):
        clean_schedule = deepcopy(schedule)
        clean_schedule['metric'] = clean_schedule.get('metric', 'clean')
        refine.test(test_dataset, clean_schedule)
        clean_run_dir = shared_run_dir

    if cfg.get('eval_poisoned', True):
        attack_loss = build_loss(cfg.get('loss', 'cross_entropy'))
        attack = build_attack(
            cfg['attack']['name'],
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=attack_loss,
            attack_kwargs=cfg['attack'].get('kwargs', {}),
            schedule=schedule,
        )
        _, poisoned_test_dataset = attack.get_poisoned_dataset()
        asr_schedule = deepcopy(schedule)
        asr_schedule['metric'] = cfg.get('poisoned_metric', 'ASR')
        refine.test(poisoned_test_dataset, asr_schedule)
        asr_run_dir = shared_run_dir

    record = build_refine_record(
        cfg,
        config_path=args.config,
        train_run_dir=train_run_dir,
        clean_run_dir=clean_run_dir,
        asr_run_dir=asr_run_dir,
    )
    append_experiment_record(record)


if __name__ == '__main__':
    main()
