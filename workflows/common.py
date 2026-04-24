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
from core.utils import accuracy, build_run_metadata, infer_num_classes, resolve_output_dir, resolve_topk, sanitize_name, write_json
from core.utils.log import Log
from workflows.refine_pipeline import build_model


ATTACK_BUILDERS = {
    'badnets': BadNets,
    'blended': Blended,
    'wanet': WaNet,
}

IMAGENET_NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _normalized_dataset_name(dataset_cfg):
    return str(dataset_cfg['name']).strip().lower()


def _dataset_image_size(dataset_cfg, default_size=None):
    image_size = dataset_cfg.get('image_size', default_size)
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return image_size
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2 and image_size[0] == image_size[1]:
        return int(image_size[0])
    return image_size


def _normalize_transform(dataset_cfg):
    normalize = dataset_cfg.get('normalize')
    if normalize is not None:
        return normalize
    if _normalized_dataset_name(dataset_cfg) in ('cub200', 'cub-200', 'cub_200'):
        return IMAGENET_NORMALIZE
    return None


def _append_normalize(transform_steps, dataset_cfg):
    normalize = _normalize_transform(dataset_cfg)
    if normalize is not None:
        transform_steps.append(transforms.Normalize(mean=normalize['mean'], std=normalize['std']))


def default_transform(dataset_cfg, train=True):
    dataset_name = dataset_cfg['name'].lower()
    transform_steps = []

    if dataset_name == 'mnist':
        image_size = dataset_cfg.get('image_size')
        if image_size is not None:
            if isinstance(image_size, int):
                image_size = [image_size, image_size]
            transform_steps.append(transforms.Resize(tuple(image_size)))
        transform_steps.append(transforms.ToTensor())
        return transforms.Compose(transform_steps)

    if dataset_name in ('cub200', 'cub-200', 'cub_200'):
        image_size = _dataset_image_size(dataset_cfg, default_size=224)
        resize_shorter = dataset_cfg.get('resize_shorter')
        if resize_shorter is None and isinstance(image_size, int):
            resize_shorter = int(round(image_size * 1.15))
        if resize_shorter is not None:
            transform_steps.append(transforms.Resize((resize_shorter, resize_shorter)))
        if train:
            transform_steps.append(transforms.RandomCrop((image_size, image_size)))
            transform_steps.append(transforms.RandomHorizontalFlip())
        else:
            transform_steps.append(transforms.CenterCrop((image_size, image_size)))
        transform_steps.append(transforms.ToTensor())
        _append_normalize(transform_steps, dataset_cfg)
        return transforms.Compose(transform_steps)

    resize_shorter = dataset_cfg.get('resize_shorter')
    if resize_shorter is not None:
        transform_steps.append(transforms.Resize(resize_shorter))

    image_size = dataset_cfg.get('image_size')
    if image_size is not None:
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        transform_steps.append(transforms.Resize(tuple(image_size)))

    transform_steps.append(transforms.ToTensor())
    _append_normalize(transform_steps, dataset_cfg)
    return transforms.Compose(transform_steps)


def load_datasets(dataset_cfg):
    dataset_name = dataset_cfg['name'].lower()
    train_transform = default_transform(dataset_cfg, train=True)
    test_transform = default_transform(dataset_cfg, train=False)

    if dataset_name == 'cifar10':
        root = dataset_cfg.get('root', './datasets')
        train_dataset = CIFAR10(root=root, train=True, download=dataset_cfg.get('download', False), transform=train_transform)
        test_dataset = CIFAR10(root=root, train=False, download=dataset_cfg.get('download', False), transform=test_transform)
    elif dataset_name == 'mnist':
        root = dataset_cfg.get('root', './datasets')
        train_dataset = MNIST(root=root, train=True, download=dataset_cfg.get('download', False), transform=train_transform)
        test_dataset = MNIST(root=root, train=False, download=dataset_cfg.get('download', False), transform=test_transform)
    elif dataset_name in ('datasetfolder', 'imagefolder', 'gtsrb', 'tiny-imagenet', 'cub200', 'cub-200', 'cub_200'):
        if 'train_dir' in dataset_cfg and 'test_dir' in dataset_cfg:
            train_dir = dataset_cfg['train_dir']
            test_dir = dataset_cfg['test_dir']
        else:
            root = dataset_cfg['root']
            train_dir = dataset_cfg.get('train_subdir', 'train')
            test_dir = dataset_cfg.get('test_subdir', 'test')
            train_dir = osp.join(root, train_dir)
            test_dir = osp.join(root, test_dir)
        train_dataset = ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    else:
        raise KeyError(f'Unsupported dataset: {dataset_cfg["name"]}')

    return train_dataset, test_dataset


def infer_in_channels(dataset):
    sample = dataset[0][0]
    if sample.ndim != 3:
        raise ValueError(f'Expected sample image with shape (C, H, W), got {tuple(sample.shape)}.')
    return int(sample.shape[0])


def infer_image_size(dataset):
    sample = dataset[0][0]
    if sample.ndim != 3:
        raise ValueError(f'Expected sample image with shape (C, H, W), got {tuple(sample.shape)}.')
    if sample.shape[-1] != sample.shape[-2]:
        raise ValueError(f'Expected square image, got {tuple(sample.shape)}.')
    return int(sample.shape[-1])


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
    model_token = sanitize_name(model_name)
    attack_token = sanitize_name(attack_name)

    schedule.setdefault('dataset_name', dataset_name)
    schedule.setdefault('model_name', model_name)
    schedule.setdefault('attack_name', attack_name)

    if defense_name is not None:
        defense_token = sanitize_name(defense_name)
        schedule.setdefault('save_dir', save_dir)
        schedule.setdefault('defense_name', defense_name)
        schedule.setdefault('experiment_name', f'{attack_token}_{model_token}_{defense_token}')
    elif attack_token != 'clean':
        schedule.setdefault('save_dir', save_dir)
        schedule.setdefault('experiment_name', f'{attack_token}_{model_token}')
    else:
        schedule.setdefault('save_dir', save_dir)
        schedule.setdefault('experiment_name', f'clean_{model_token}_clean')

    if schedule.get('device') == 'GPU':
        schedule.setdefault('GPU_num', 1)
    return schedule


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _find_insert_index(dataset):
    transform = getattr(dataset, 'transform', None)
    if transform is None or not hasattr(transform, 'transforms'):
        return 0
    for index, item in enumerate(transform.transforms):
        if isinstance(item, transforms.ToTensor):
            return index
    return len(transform.transforms)


def resolve_poisoned_transform_indices(train_dataset, test_dataset, attack_kwargs):
    train_index = attack_kwargs.get('poisoned_transform_train_index')
    if train_index is None:
        train_index = _find_insert_index(train_dataset)
    test_index = attack_kwargs.get('poisoned_transform_test_index')
    if test_index is None:
        test_index = _find_insert_index(test_dataset)
    target_index = attack_kwargs.get('poisoned_target_transform_index')
    if target_index is None:
        target_index = 0
    return train_index, test_index, target_index


def build_badnets_pattern(image_size, trigger_size=3, alpha=1.0):
    pattern = torch.zeros((image_size, image_size), dtype=torch.uint8)
    pattern[-trigger_size:, -trigger_size:] = 255
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[-trigger_size:, -trigger_size:] = alpha
    return pattern, weight


def default_trigger_size(image_size):
    if image_size <= 32:
        return 3
    return max(3, int(round(image_size * 0.08)))


def gen_wanet_grid(height, k=4):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.interpolate(ins, size=height, mode='bicubic', align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid


def prepare_attack_kwargs(name, train_dataset, test_dataset, attack_kwargs):
    prepared = deepcopy(attack_kwargs)
    prepared.setdefault('y_target', 0)
    prepared.setdefault('poisoned_rate', 0.1)

    train_index, test_index, target_index = resolve_poisoned_transform_indices(
        train_dataset,
        test_dataset,
        prepared,
    )
    prepared['poisoned_transform_train_index'] = train_index
    prepared['poisoned_transform_test_index'] = test_index
    prepared['poisoned_target_transform_index'] = target_index

    image_size = infer_image_size(train_dataset)
    key = name.lower()

    if key in ('badnets', 'blended'):
        trigger_size = int(prepared.pop('trigger_size', default_trigger_size(image_size)))
        alpha = 1.0 if key == 'badnets' else float(prepared.pop('blended_alpha', 0.2))
        if prepared.get('pattern') is None or prepared.get('weight') is None:
            pattern, weight = build_badnets_pattern(image_size=image_size, trigger_size=trigger_size, alpha=alpha)
            prepared['pattern'] = pattern
            prepared['weight'] = weight

    if key == 'wanet':
        grid_k = int(prepared.pop('grid_k', 4))
        prepared.setdefault('noise', True)
        if prepared.get('identity_grid') is None or prepared.get('noise_grid') is None:
            identity_grid, noise_grid = gen_wanet_grid(height=image_size, k=grid_k)
            prepared['identity_grid'] = identity_grid
            prepared['noise_grid'] = noise_grid

    return prepared


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
        if epoch in schedule.get('schedule', []):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= schedule.get('gamma', 0.1)

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
                f"lr:{optimizer.param_groups[0]['lr']}, "
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
    attack_kwargs = prepare_attack_kwargs(
        key,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        attack_kwargs=attack_kwargs,
    )
    attack = attack_cls(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss=loss,
        schedule=schedule,
        **attack_kwargs,
    )
    attack.prepared_attack_kwargs = deepcopy(attack_kwargs)
    return attack


def build_model_from_config(model_cfg, num_classes):
    model = build_model(
        model_cfg['name'],
        num_classes=num_classes,
        **model_cfg.get('kwargs', {}),
    )
    return maybe_load_weights(model, model_cfg.get('checkpoint'))
