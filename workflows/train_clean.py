import argparse

from workflows.common import (
    build_loss,
    build_model_from_config,
    complete_schedule,
    infer_num_classes,
    load_datasets,
    load_json,
    set_global_seed,
    train_clean_model,
)


def main():
    parser = argparse.ArgumentParser(description='Train a clean model.')
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
        attack_name='clean',
    )

    train_clean_model(model, train_dataset, test_dataset, loss, schedule)


if __name__ == '__main__':
    main()
