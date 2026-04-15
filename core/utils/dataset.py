from copy import deepcopy

import numpy as np
import torch


def clone_dataset(dataset):
    return deepcopy(dataset)


def infer_num_classes(dataset):
    if hasattr(dataset, 'classes') and dataset.classes is not None:
        return len(dataset.classes)

    labels = get_dataset_labels(dataset)
    if labels is None:
        raise ValueError('Unable to infer the number of classes from the dataset.')

    return len(set(labels))


def get_dataset_labels(dataset):
    if hasattr(dataset, 'targets'):
        return _to_label_list(dataset.targets)
    if hasattr(dataset, 'labels'):
        return _to_label_list(dataset.labels)
    if hasattr(dataset, 'samples'):
        return [sample[1] for sample in dataset.samples]
    return None


def filter_dataset_excluding_label(dataset, excluded_label):
    filtered_dataset = clone_dataset(dataset)
    labels = get_dataset_labels(filtered_dataset)
    if labels is None:
        raise NotImplementedError('Dataset filtering requires targets, labels, or samples attributes.')

    kept_indices = [idx for idx, label in enumerate(labels) if int(label) != int(excluded_label)]
    _filter_dataset_in_place(filtered_dataset, kept_indices)
    return filtered_dataset


def resolve_topk(output, topk=(1, 5)):
    if output.ndim != 2:
        raise ValueError(f'Expected classification logits with shape (N, C), got {tuple(output.shape)}.')

    num_classes = output.size(1)
    effective_topk = []
    for k in topk:
        capped_k = min(int(k), num_classes)
        if capped_k not in effective_topk:
            effective_topk.append(capped_k)
    return tuple(effective_topk)


def _filter_dataset_in_place(dataset, kept_indices):
    if hasattr(dataset, 'data'):
        dataset.data = _slice_sequence(dataset.data, kept_indices)

    if hasattr(dataset, 'targets'):
        dataset.targets = _slice_sequence(dataset.targets, kept_indices)

    if hasattr(dataset, 'labels'):
        dataset.labels = _slice_sequence(dataset.labels, kept_indices)

    if hasattr(dataset, 'samples'):
        dataset.samples = [dataset.samples[idx] for idx in kept_indices]

    if hasattr(dataset, 'imgs'):
        dataset.imgs = [dataset.imgs[idx] for idx in kept_indices]


def _slice_sequence(values, indices):
    if torch.is_tensor(values):
        return values[indices]
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[idx] for idx in indices]


def _to_label_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)
