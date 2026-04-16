import argparse

from workflows.common import (
    build_attack,
    build_loss,
    build_model_from_config,
    complete_schedule,
    infer_num_classes,
    load_datasets,
    load_json,
    set_global_seed,
)
from workflows.reporting import (
    append_experiment_record,
    build_attack_record,
    capture_run_dirs,
    detect_created_run_dir,
)


def main():
    parser = argparse.ArgumentParser(description='Run a backdoor attack workflow.')
    parser.add_argument('--config', required=True, help='Path to a JSON config file.')
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_global_seed(cfg.get('seed', 0))

    train_dataset, test_dataset = load_datasets(cfg['dataset'])
    num_classes = cfg.get('num_classes', infer_num_classes(train_dataset))
    model = build_model_from_config(cfg['model'], num_classes)
    loss = build_loss(cfg.get('loss', 'cross_entropy'))

    schedule = complete_schedule(
        cfg['schedule'],
        dataset_name=cfg['dataset']['name'],
        model_name=cfg['model']['name'],
        attack_name=cfg['attack']['name'],
    )
    schedule.setdefault('benign_training', False)

    attack = build_attack(
        cfg['attack']['name'],
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss=loss,
        attack_kwargs=cfg['attack'].get('kwargs', {}),
        schedule=schedule,
    )

    train_run_dir = None
    if cfg.get('run_train', True):
        train_before = capture_run_dirs(schedule, stage='attacks', method_name=cfg['attack']['name'])
        attack.train(schedule)
        train_run_dir = detect_created_run_dir(
            train_before,
            schedule,
            stage='attacks',
            method_name=cfg['attack']['name'],
        )
    if cfg.get('run_test', True):
        attack.test(schedule)

    if train_run_dir:
        record = build_attack_record(cfg, config_path=args.config, train_run_dir=train_run_dir)
        append_experiment_record(record)


if __name__ == '__main__':
    main()
