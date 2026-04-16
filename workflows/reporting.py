import csv
import json
import os
import os.path as osp
import time

from core.utils import sanitize_name


REPORT_CSV_PATH = osp.join('outputs', 'reports', 'experiment_results.csv')


def _run_prefix(schedule, stage='defenses', method_name=None, extra_tag=None):
    del stage
    attack_name = sanitize_name(schedule.get('attack_name'))
    model_name = sanitize_name(schedule.get('model_name'))
    method_name = sanitize_name(method_name or schedule.get('defense_name') or 'unknown')
    parts = [attack_name, model_name, method_name]
    if extra_tag is not None:
        parts.append(sanitize_name(extra_tag))
    return '_'.join(parts) + '_'


def _stage_dataset_dir(schedule, stage='defenses'):
    return osp.join(
        schedule.get('save_dir', 'outputs'),
        sanitize_name(stage),
        sanitize_name(schedule.get('dataset_name')),
    )


def list_matching_run_dirs(schedule, stage='defenses', method_name=None, extra_tag=None):
    root = _stage_dataset_dir(schedule, stage=stage)
    if not osp.isdir(root):
        return []

    prefix = _run_prefix(schedule, stage=stage, method_name=method_name, extra_tag=extra_tag)
    candidates = []
    for name in os.listdir(root):
        path = osp.join(root, name)
        if osp.isdir(path) and name.startswith(prefix):
            candidates.append(path)
    candidates.sort(key=lambda item: osp.getmtime(item))
    return candidates


def capture_run_dirs(schedule, stage='defenses', method_name=None, extra_tag=None):
    return set(list_matching_run_dirs(schedule, stage=stage, method_name=method_name, extra_tag=extra_tag))


def detect_created_run_dir(previous, schedule, stage='defenses', method_name=None, extra_tag=None):
    current = list_matching_run_dirs(schedule, stage=stage, method_name=method_name, extra_tag=extra_tag)
    new_dirs = [path for path in current if path not in previous]
    if new_dirs:
        return new_dirs[-1]
    if current:
        return current[-1]
    return None


def load_json_if_exists(path):
    if path is None or not osp.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metrics_from_run_dir(run_dir):
    if run_dir is None:
        return None
    return load_json_if_exists(osp.join(run_dir, 'metrics.json'))


def append_experiment_record(record, csv_path=REPORT_CSV_PATH):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    fieldnames = [
        'recorded_at',
        'config_path',
        'dataset',
        'attack',
        'model',
        'defense',
        'attack_checkpoint',
        'train_epochs',
        'batch_size',
        'lr',
        'train_run_dir',
        'clean_run_dir',
        'asr_run_dir',
        'clean_top1_accuracy',
        'clean_topk_accuracy',
        'asr_top1_accuracy',
        'asr_topk_accuracy',
        'clean_metric_name',
        'asr_metric_name',
        'train_loss',
        'defense_kwargs_json',
        'attack_kwargs_json',
    ]

    exists = osp.isfile(csv_path)
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {key: record.get(key, '') for key in fieldnames}
        writer.writerow(row)


def _parse_attack_log_metrics(run_dir):
    log_path = osp.join(run_dir, 'log.txt')
    if not osp.isfile(log_path):
        return {}

    benign = {}
    poisoned = {}
    last_train_loss = ''
    current_stage = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if '==========Test result on benign test dataset==========' in line:
                current_stage = 'benign'
                continue
            if '==========Test result on poisoned test dataset==========' in line:
                current_stage = 'poisoned'
                continue
            if 'Epoch:' in line and 'loss:' in line:
                try:
                    last_train_loss = float(line.split('loss:')[1].split(',')[0].strip())
                except Exception:
                    pass
            if 'Top-1 correct / Total:' in line and 'Top-1 accuracy:' in line:
                try:
                    top1 = float(line.split('Top-1 accuracy:')[1].split(',')[0].strip())
                    topk_label = line.split('Top-5 correct / Total:')[0]
                    if 'Top-5 accuracy:' in line:
                        topk = float(line.split('Top-5 accuracy:')[1].split(',')[0].strip())
                        topk_eval = 5
                    else:
                        topk = ''
                        topk_eval = ''
                except Exception:
                    current_stage = None
                    continue
                target = benign if current_stage == 'benign' else poisoned if current_stage == 'poisoned' else None
                if target is not None:
                    target['top1_accuracy'] = top1
                    target['topk_accuracy'] = topk
                    target['topk_eval'] = topk_eval
                current_stage = None

    return {
        'benign': benign,
        'poisoned': poisoned,
        'train_loss': last_train_loss,
    }


def build_refine_record(cfg, config_path, train_run_dir=None, clean_run_dir=None, asr_run_dir=None):
    train_metrics = load_metrics_from_run_dir(train_run_dir)
    clean_metrics = load_metrics_from_run_dir(clean_run_dir)
    asr_metrics = load_metrics_from_run_dir(asr_run_dir)

    clean_topk_key = None
    asr_topk_key = None
    if clean_metrics is not None:
        clean_topk_key = next((key for key in clean_metrics if key.startswith('top') and key.endswith('_accuracy') and key != 'top1_accuracy'), None)
    if asr_metrics is not None:
        asr_topk_key = next((key for key in asr_metrics if key.startswith('top') and key.endswith('_accuracy') and key != 'top1_accuracy'), None)

    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': cfg['attack']['name'],
        'model': cfg['model']['name'],
        'defense': cfg['refine'].get('defense_name', 'refine'),
        'attack_checkpoint': cfg['model'].get('checkpoint', ''),
        'train_epochs': cfg['schedule'].get('epochs', ''),
        'batch_size': cfg['schedule'].get('batch_size', ''),
        'lr': cfg['schedule'].get('lr', ''),
        'train_run_dir': train_run_dir or '',
        'clean_run_dir': clean_run_dir or '',
        'asr_run_dir': asr_run_dir or '',
        'clean_top1_accuracy': '' if clean_metrics is None else clean_metrics.get('top1_accuracy', ''),
        'clean_topk_accuracy': '' if clean_metrics is None or clean_topk_key is None else clean_metrics.get(clean_topk_key, ''),
        'asr_top1_accuracy': '' if asr_metrics is None else asr_metrics.get('top1_accuracy', ''),
        'asr_topk_accuracy': '' if asr_metrics is None or asr_topk_key is None else asr_metrics.get(asr_topk_key, ''),
        'clean_metric_name': '' if clean_metrics is None else clean_metrics.get('metric', ''),
        'asr_metric_name': '' if asr_metrics is None else asr_metrics.get('metric', ''),
        'train_loss': '' if train_metrics is None else train_metrics.get('loss', ''),
        'defense_kwargs_json': json.dumps(cfg['refine'].get('defense_kwargs', {}), ensure_ascii=False, sort_keys=True),
        'attack_kwargs_json': json.dumps(cfg['attack'].get('kwargs', {}), ensure_ascii=False, sort_keys=True),
    }


def build_clean_record(cfg, config_path, train_run_dir=None):
    train_metrics = load_metrics_from_run_dir(train_run_dir)
    clean_topk_key = None
    if train_metrics is not None:
        clean_topk_key = next(
            (
                key
                for key in train_metrics
                if key.startswith('top') and key.endswith('_accuracy') and key != 'top1_accuracy'
            ),
            None,
        )

    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': 'clean',
        'model': cfg['model']['name'],
        'defense': 'clean',
        'attack_checkpoint': cfg['model'].get('checkpoint', ''),
        'train_epochs': cfg['schedule'].get('epochs', ''),
        'batch_size': cfg['schedule'].get('batch_size', ''),
        'lr': cfg['schedule'].get('lr', ''),
        'train_run_dir': train_run_dir or '',
        'clean_run_dir': train_run_dir or '',
        'asr_run_dir': '',
        'clean_top1_accuracy': '' if train_metrics is None else train_metrics.get('top1_accuracy', ''),
        'clean_topk_accuracy': '' if train_metrics is None or clean_topk_key is None else train_metrics.get(clean_topk_key, ''),
        'asr_top1_accuracy': '',
        'asr_topk_accuracy': '',
        'clean_metric_name': 'clean',
        'asr_metric_name': '',
        'train_loss': '' if train_metrics is None else train_metrics.get('train_loss', ''),
        'defense_kwargs_json': json.dumps({}, ensure_ascii=False, sort_keys=True),
        'attack_kwargs_json': json.dumps({}, ensure_ascii=False, sort_keys=True),
    }


def build_attack_record(cfg, config_path, train_run_dir=None):
    parsed = _parse_attack_log_metrics(train_run_dir) if train_run_dir else {}
    benign = parsed.get('benign', {})
    poisoned = parsed.get('poisoned', {})

    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': cfg['attack']['name'],
        'model': cfg['model']['name'],
        'defense': 'attack',
        'attack_checkpoint': cfg['model'].get('checkpoint', ''),
        'train_epochs': cfg['schedule'].get('epochs', ''),
        'batch_size': cfg['schedule'].get('batch_size', ''),
        'lr': cfg['schedule'].get('lr', ''),
        'train_run_dir': train_run_dir or '',
        'clean_run_dir': '',
        'asr_run_dir': '',
        'clean_top1_accuracy': benign.get('top1_accuracy', ''),
        'clean_topk_accuracy': benign.get('topk_accuracy', ''),
        'asr_top1_accuracy': poisoned.get('top1_accuracy', ''),
        'asr_topk_accuracy': poisoned.get('topk_accuracy', ''),
        'clean_metric_name': 'benign_test',
        'asr_metric_name': 'poisoned_test',
        'train_loss': parsed.get('train_loss', ''),
        'defense_kwargs_json': json.dumps({}, ensure_ascii=False, sort_keys=True),
        'attack_kwargs_json': json.dumps(cfg['attack'].get('kwargs', {}), ensure_ascii=False, sort_keys=True),
    }
