import csv
import json
import os
import os.path as osp
import time

from core.utils import sanitize_name


REPORT_CSV_PATH = osp.join('outputs', 'reports', 'experiment_summary.csv')


def _run_prefix(schedule, stage='defenses', method_name=None, extra_tag=None):
    stage_name = sanitize_name(stage)
    attack_name = sanitize_name(schedule.get('attack_name'))
    model_name = sanitize_name(schedule.get('model_name'))
    method_name = sanitize_name(method_name or schedule.get('defense_name') or 'unknown')
    if stage_name == 'clean':
        parts = [model_name, 'clean']
    elif stage_name == 'attacks':
        parts = [attack_name, model_name]
    elif stage_name == 'defenses':
        parts = [attack_name, model_name, method_name]
    else:
        parts = [method_name, model_name]
    if extra_tag is not None:
        parts.append(sanitize_name(extra_tag))
    return '_'.join(parts) + '_'


def _stage_dataset_dir(schedule, stage='defenses'):
    return osp.join(
        schedule.get('save_dir', 'outputs'),
        sanitize_name(schedule.get('dataset_name')),
        sanitize_name(stage),
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


def load_metrics_from_run_dir(run_dir, filename='metrics.json'):
    if run_dir is None:
        return None
    return load_json_if_exists(osp.join(run_dir, filename))


def _output_checkpoint(run_dir, epochs=None, prefix='ckpt_epoch_'):
    if run_dir is None or not osp.isdir(run_dir):
        return ''
    if epochs not in (None, ''):
        candidate = osp.join(run_dir, f'{prefix}{epochs}.pth')
        if osp.isfile(candidate):
            return candidate.replace('\\', '/')

    checkpoints = []
    for name in os.listdir(run_dir):
        if not (name.startswith(prefix) and name.endswith('.pth')):
            continue
        epoch_text = name[len(prefix):-4]
        try:
            epoch = int(epoch_text)
        except ValueError:
            continue
        checkpoints.append((epoch, osp.join(run_dir, name)))
    if not checkpoints:
        return ''
    checkpoints.sort()
    return checkpoints[-1][1].replace('\\', '/')


def append_experiment_record(record, csv_path=REPORT_CSV_PATH):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    fieldnames = [
        'recorded_at',
        'stage',
        'config_path',
        'dataset',
        'attack',
        'model',
        'defense',
        'clean_acc',
        'asr',
        'train_loss',
        'epochs',
        'seed',
        'run_dir',
        'checkpoint',
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
    clean_metric_name = cfg['schedule'].get('metric', 'clean')
    asr_metric_name = cfg.get('poisoned_metric', 'ASR')
    clean_metrics_file = f"metrics_{sanitize_name(clean_metric_name)}.json"
    asr_metrics_file = f"metrics_{sanitize_name(asr_metric_name)}.json"

    train_metrics = load_metrics_from_run_dir(train_run_dir, 'metrics_train_unet.json')
    if train_metrics is None:
        train_metrics = load_metrics_from_run_dir(train_run_dir)
    clean_metrics = load_metrics_from_run_dir(clean_run_dir, clean_metrics_file)
    if clean_metrics is None:
        clean_metrics = load_metrics_from_run_dir(clean_run_dir)
    asr_metrics = load_metrics_from_run_dir(asr_run_dir, asr_metrics_file)
    if asr_metrics is None:
        asr_metrics = load_metrics_from_run_dir(asr_run_dir)

    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'stage': 'defense',
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': cfg['attack']['name'],
        'model': cfg['model']['name'],
        'defense': cfg['refine'].get('defense_name', 'refine'),
        'clean_acc': '' if clean_metrics is None else clean_metrics.get('top1_accuracy', ''),
        'asr': '' if asr_metrics is None else asr_metrics.get('top1_accuracy', ''),
        'train_loss': '' if train_metrics is None else train_metrics.get('loss', ''),
        'epochs': cfg['schedule'].get('epochs', ''),
        'seed': cfg.get('seed', ''),
        'run_dir': train_run_dir or clean_run_dir or asr_run_dir or '',
        'checkpoint': _output_checkpoint(train_run_dir, cfg['schedule'].get('epochs', '')),
    }


def build_clean_record(cfg, config_path, train_run_dir=None):
    train_metrics = load_metrics_from_run_dir(train_run_dir)
    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'stage': 'clean',
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': 'clean',
        'model': cfg['model']['name'],
        'defense': 'clean',
        'clean_acc': '' if train_metrics is None else train_metrics.get('top1_accuracy', ''),
        'asr': '',
        'train_loss': '' if train_metrics is None else train_metrics.get('train_loss', ''),
        'epochs': cfg['schedule'].get('epochs', ''),
        'seed': cfg.get('seed', ''),
        'run_dir': train_run_dir or '',
        'checkpoint': _output_checkpoint(train_run_dir, cfg['schedule'].get('epochs', '')),
    }


def build_attack_record(cfg, config_path, train_run_dir=None):
    parsed = _parse_attack_log_metrics(train_run_dir) if train_run_dir else {}
    benign = parsed.get('benign', {})
    poisoned = parsed.get('poisoned', {})

    return {
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'stage': 'attack',
        'config_path': config_path,
        'dataset': cfg['dataset']['name'],
        'attack': cfg['attack']['name'],
        'model': cfg['model']['name'],
        'defense': 'none',
        'clean_acc': benign.get('top1_accuracy', ''),
        'asr': poisoned.get('top1_accuracy', ''),
        'train_loss': parsed.get('train_loss', ''),
        'epochs': cfg['schedule'].get('epochs', ''),
        'seed': cfg.get('seed', ''),
        'run_dir': train_run_dir or '',
        'checkpoint': _output_checkpoint(train_run_dir, cfg['schedule'].get('epochs', '')),
    }
