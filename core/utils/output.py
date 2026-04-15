import json
import os
import os.path as osp
import re
import time
from copy import deepcopy


def sanitize_name(value, default='unknown'):
    if value is None:
        return default

    text = str(value).strip().lower()
    if not text:
        return default

    text = re.sub(r'[^a-zA-Z0-9._-]+', '-', text)
    text = re.sub(r'-{2,}', '-', text).strip('-')
    return text or default


def resolve_output_dir(schedule, stage, method_name=None, extra_tag=None):
    schedule = schedule if schedule is not None else {}
    base_dir = schedule.get('save_dir', 'outputs')
    dataset_name = sanitize_name(schedule.get('dataset_name'))
    attack_name = sanitize_name(schedule.get('attack_name'))
    model_name = sanitize_name(schedule.get('model_name'))
    method_name = sanitize_name(method_name or schedule.get('defense_name') or stage)
    extra_tag = sanitize_name(extra_tag, default='') if extra_tag is not None else ''

    timestamp = schedule.setdefault('run_timestamp', time.strftime('%Y%m%d_%H%M%S', time.localtime()))

    run_name_parts = [attack_name, model_name, method_name]
    if extra_tag:
        run_name_parts.append(extra_tag)
    run_name_parts.append(timestamp)
    run_name = '_'.join(run_name_parts)

    work_dir = osp.join(base_dir, sanitize_name(stage), dataset_name, run_name)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_run_metadata(schedule, **extra):
    payload = deepcopy(schedule) if schedule is not None else {}
    payload.update(extra)
    return payload
