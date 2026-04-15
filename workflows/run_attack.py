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

    if cfg.get('run_train', True):
        attack.train(schedule)
    if cfg.get('run_test', True):
        attack.test(schedule)


if __name__ == '__main__':
    main()
