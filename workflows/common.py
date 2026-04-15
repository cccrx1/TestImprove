import json
import os
import os.path as osp
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageFolder

from core.attacks import BadNets, Blended, WaNet
from core.utils import accuracy, build_run_metadata, infer_num_classes, resolve_output_dir, resolve_topk, write_json
from core.utils.log import Log
from workflows.refine_pipeline import build_model


ATTACK_BUILDERS = {
    'badnets': BadNets,
    'blended': Blended,
    'wanet': WaNet,
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def default_transform(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose([transforms.ToTensor()])


def load_datasets(dataset_cfg):
    dataset_name = dataset_cfg['name'].lower()
    transform = default_transform(dataset_name)

    if dataset_name == 'cifar10':
        root = dataset_cfg.get('root', './datasets')
        train_dataset = CIFAR10(root=root, train=True, download=dataset_cfg.get('download', False), transform=transform)
        test_dataset = CIFAR10(root=root, train=False, download=dataset_cfg.get('download', False), transform=transform)
    elif dataset_name == 'mnist':
        root = dataset_cfg.get('root', './datasets')
        train_dataset = MNIST(root=root, train=True, download=dataset_cfg.get('download', False), transform=transform)
        test_dataset = MNIST(root=root, train=False, download=dataset_cfg.get('download', False), transform=transform)
    elif dataset_name in ('datasetfolder', 'imagefolder', 'gtsrb', 'tiny-imagenet'):
        train_dir = dataset_cfg['train_dir']
        test_dir = dataset_cfg['test_dir']
        train_dataset = ImageFolder(root=train_dir, transform=transform)
        test_dataset = ImageFolder(root=test_dir, transform=transform)
    else:
        raise KeyError(f'Unsupported dataset: {dataset_cfg["name"]}')

    return train_dataset, test_dataset


def infer_in_channels(dataset):
    sample = dataset[0][0]
    if sample.ndim != 3:
        raise ValueError(f'Expected sample image with shape (C, H, W), got {tuple(sample.shape)}.')
    return int(sample.shape[0])


def build_loss(loss_name):
    key = loss_name.lower()
    if key in ('ce', 'cross_entropy', 'crossentropyloss'):
        return nn.CrossEntropyLoss()
    raise KeyError(f'Unsupported loss: {loss_name}')


def maybe_load_weights(model, ckpt_path):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    return model


def complete_schedule(schedule, dataset_name, model_name, attack_name='clean', defense_name=None, save_dir='outputs'):
    schedule = deepcopy(schedule)
    schedule.setdefault('save_dir', save_dir)
    schedule.setdefault('dataset_name', dataset_name)
    schedule.setdefault('model_name', model_name)
    schedule.setdefault('attack_name', attack_name)
    if defense_name is not None:
        schedule.setdefault('defense_name', defense_name)
    return schedule


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_clean_model(model, train_dataset, test_dataset, loss, schedule):
    schedule = deepcopy(schedule)
    work_dir = resolve_output_dir(schedule, stage='clean', method_name='clean')
    log = Log(osp.join(work_dir, 'log.txt'))
    write_json(
        osp.join(work_dir, 'config.json'),
        build_run_metadata(schedule, stage='clean', output_dir=work_dir),
    )

    if schedule.get('device') == 'GPU':
        if 'CUDA_VISIBLE_DEVICES' in schedule:
            os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_loader = DataLoader(
        train_dataset,
        batch_size=schedule['batch_size'],
        shuffle=True,
        num_workers=schedule['num_workers'],
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=schedule['batch_size'],
        shuffle=False,
        num_workers=schedule['num_workers'],
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=schedule['lr'],
        momentum=schedule.get('momentum', 0.9),
        weight_decay=schedule.get('weight_decay', 5e-4),
    )

    last_metrics = {}
    for epoch in range(schedule['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            batch_loss = loss(logits, labels)
            batch_loss.backward()
            optimizer.step()
            running_loss += float(batch_loss)

        if (epoch + 1) % schedule.get('test_epoch_interval', 1) == 0:
            metrics = evaluate_classifier(model, test_loader, device)
            metrics.update({
                'epoch': epoch + 1,
                'train_loss': running_loss / max(len(train_loader), 1),
                'stage': 'clean',
            })
            last_metrics = metrics
            log(
                f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}] "
                f"Epoch:{epoch+1}/{schedule['epochs']}, "
                f"train_loss:{metrics['train_loss']}, top1:{metrics['top1_accuracy']}, top{metrics['topk_eval']}:{metrics['topk_accuracy']}\n"
            )
            write_json(osp.join(work_dir, 'metrics.json'), metrics)

        if (epoch + 1) % schedule.get('save_epoch_interval', 1) == 0:
            torch.save(model.cpu().state_dict(), osp.join(work_dir, f'ckpt_epoch_{epoch+1}.pth'))
            model = model.to(device)

    return work_dir, last_metrics


def evaluate_classifier(model, data_loader, device):
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits_list.append(model(images.to(device)).cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    total_num = labels.size(0)
    topk = resolve_topk(logits, topk=(1, 5))
    metrics = accuracy(logits, labels, topk=topk)
    metric_map = dict(zip(topk, metrics))
    top1 = float(metric_map[1].item() / 100.0)
    topk_eval = topk[-1]
    topk_accuracy = float(metric_map[topk_eval].item() / 100.0)
    return {
        'total_num': int(total_num),
        'top1_accuracy': top1,
        'topk_eval': int(topk_eval),
        'topk_accuracy': topk_accuracy,
    }


def build_attack(name, train_dataset, test_dataset, model, loss, attack_kwargs, schedule):
    key = name.lower()
    if key not in ATTACK_BUILDERS:
        supported = ', '.join(sorted(ATTACK_BUILDERS))
        raise KeyError(f'Unsupported attack: {name}. Supported: {supported}.')
    attack_cls = ATTACK_BUILDERS[key]
    return attack_cls(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss=loss,
        schedule=schedule,
        **attack_kwargs,
    )


def build_model_from_config(model_cfg, num_classes):
    model = build_model(
        model_cfg['name'],
        num_classes=num_classes,
        **model_cfg.get('kwargs', {}),
    )
    return maybe_load_weights(model, model_cfg.get('checkpoint'))
