'''
REFINE with prediction-distribution consistency distillation.

This variant keeps the original REFINE objective while adding a KL-based
distillation term between the frozen model's predictions on clean inputs
and purified inputs.
'''

import os
import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .REFINE import REFINE
from ..utils import (
    Log,
    SupConLoss,
    accuracy,
    build_run_metadata,
    infer_num_classes,
    resolve_output_dir,
    resolve_topk,
    write_json,
)


class REFINE_KD(REFINE):
    """REFINE with distribution-consistency distillation."""

    def __init__(
        self,
        unet,
        model,
        pretrain=None,
        arr_path=None,
        num_classes=10,
        lmd=0.1,
        seed=0,
        deterministic=False,
        kd_weight=0.05,
        kd_temperature=2.0,
    ):
        super(REFINE_KD, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic,
        )
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)

    def _distillation_loss(self, clean_logits, purified_logits):
        temperature = self.kd_temperature
        teacher_prob = F.softmax(clean_logits / temperature, dim=1)
        student_log_prob = F.log_softmax(purified_logits / temperature, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)

    def _test(
        self,
        dataset,
        device,
        batch_size=16,
        num_workers=8,
        loss_func=torch.nn.BCELoss(reduction='none'),
        supconloss_func=SupConLoss(),
    ):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)

            losses = []
            for batch_img, _ in test_loader:
                batch_img = batch_img.to(device)

                bsz = batch_img.shape[0]
                clean_logits = self.model(batch_img)
                f_index = clean_logits.argmax(1)
                f_label = torch.zeros_like(clean_logits).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                features = self.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                features = torch.cat([features.unsqueeze(1), features.unsqueeze(1)], dim=1)
                supconloss = supconloss_func(features, f_index)
                kd_loss = self._distillation_loss(clean_logits, self.Y_adv)

                loss = loss_func(logit, f_label).mean() + self.lmd * supconloss + self.kd_weight * kd_loss
                losses.append(loss.detach().cpu().view(1))

            losses = torch.cat(losses, dim=0)
            return losses.mean()

    def train_unet(self, train_dataset, test_dataset, schedule):
        schedule = deepcopy(schedule)
        schedule.setdefault('defense_name', 'refine_kd')

        if self.num_classes is None:
            self.num_classes = infer_num_classes(train_dataset)
            self.init_label_shuffle()

        if 'pretrain' in schedule:
            self.unet.load_state_dict(torch.load(schedule['pretrain']), strict=False)
        if 'arr_path' in schedule:
            self.arr_shuffle = np.array(torch.load(schedule['arr_path']))

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device('cuda:0')
            else:
                gpus = list(range(schedule['GPU_num']))
                self.unet = nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
                device = torch.device(f'cuda:{gpus[0]}')
        else:
            device = torch.device('cpu')

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker,
        )

        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='mean')
        supconloss_func = SupConLoss()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),
            schedule['lr'],
            schedule['betas'],
            schedule['eps'],
            schedule['weight_decay'],
            schedule['amsgrad'],
        )

        work_dir = resolve_output_dir(schedule, stage='defenses', method_name='refine_kd')
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, osp.join(work_dir, 'label_shuffle.pth'))
        write_json(
            osp.join(work_dir, 'config.json'),
            build_run_metadata(
                schedule,
                stage='defenses',
                defense_name='refine_kd',
                output_dir=work_dir,
                num_classes=self.num_classes,
                kd_weight=self.kd_weight,
                kd_temperature=self.kd_temperature,
            ),
        )

        iteration = 0
        last_time = time.time()

        msg = (
            f"Total train samples: {len(train_dataset)}\n"
            f"Total test samples: {len(test_dataset)}\n"
            f"Batch size: {schedule['batch_size']}\n"
            f"iteration every epoch: {len(train_dataset) // schedule['batch_size']}\n"
            f"Initial learning rate: {schedule['lr']}\n"
            f"KD weight: {self.kd_weight}\n"
            f"KD temperature: {self.kd_temperature}\n"
        )
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']

            for batch_id, (batch_img, _) in enumerate(train_loader):
                batch_img = batch_img.to(device)

                bsz = batch_img.shape[0]
                clean_logits = self.model(batch_img)
                f_index = clean_logits.argmax(1)
                f_label = torch.zeros_like(clean_logits).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                features = self.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                features = torch.cat([features.unsqueeze(1), features.unsqueeze(1)], dim=1)
                supconloss = supconloss_func(features, f_index)
                ce_loss = loss_func(logit, f_label)
                kd_loss = self._distillation_loss(clean_logits, self.Y_adv)
                loss = ce_loss + self.lmd * supconloss + self.kd_weight * kd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = (
                        time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) +
                        f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_loader)}, "
                        f"lr: {schedule['lr']}, loss: {float(loss)}, ce: {float(ce_loss)}, "
                        f"supcon: {float(supconloss)}, kd: {float(kd_loss)}, time: {time.time()-last_time}\n"
                    )
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                metrics = {
                    'loss': float(loss),
                    'epoch': i + 1,
                    'stage': 'defenses',
                    'defense_name': 'refine_kd',
                    'metric': 'train_unet_test_loss',
                    'kd_weight': self.kd_weight,
                    'kd_temperature': self.kd_temperature,
                }
                write_json(osp.join(work_dir, 'metrics.json'), metrics)
                msg = (
                    '==========Test result on test dataset==========\n' +
                    time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) +
                    f"loss: {loss}, time: {time.time()-last_time}\n"
                )
                log(msg)

                self.unet = self.unet.to(device)
                self.unet.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.unet.eval()
                self.unet = self.unet.cpu()
                ckpt_unet_path = osp.join(work_dir, f'ckpt_epoch_{i+1}.pth')
                torch.save(self.unet.state_dict(), ckpt_unet_path)
                self.unet = self.unet.to(device)
                self.unet.train()

    def test(self, dataset, schedule):
        schedule = deepcopy(schedule)
        schedule.setdefault('defense_name', 'refine_kd')
        work_dir = resolve_output_dir(
            schedule,
            stage='defenses',
            method_name='refine_kd',
            extra_tag=schedule.get('metric'),
        )
        log = Log(osp.join(work_dir, 'log.txt'))
        write_json(
            osp.join(work_dir, 'config.json'),
            build_run_metadata(
                schedule,
                stage='defenses',
                defense_name='refine_kd',
                output_dir=work_dir,
                num_classes=self.num_classes,
                kd_weight=self.kd_weight,
                kd_temperature=self.kd_temperature,
            ),
        )

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device('cuda:0')
            else:
                gpus = list(range(schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                device = torch.device(f'cuda:{gpus[0]}')
        else:
            device = torch.device('cpu')

        last_time = time.time()
        batch_size = schedule.get('batch_size', 16)
        num_workers = schedule.get('num_workers', 8)
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
            )

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)
            self.model.eval()

            predict_digits = []
            labels = []
            for batch_img, batch_label in test_loader:
                batch_img = batch_img.to(device)
                predict_digits.append(self.forward(batch_img).cpu())
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)

            total_num = labels.size(0)
            topk = resolve_topk(predict_digits, topk=(1, 5))
            metrics = accuracy(predict_digits, labels, topk=topk)
            metric_map = dict(zip(topk, metrics))
            prec1 = metric_map[1]
            topk_eval = topk[-1]
            preck = metric_map[topk_eval]
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            topk_correct = int(round(preck.item() / 100.0 * total_num))
            metrics = {
                'stage': 'defenses',
                'defense_name': 'refine_kd',
                'metric': schedule['metric'],
                'top1_correct': top1_correct,
                'top1_accuracy': top1_correct / total_num,
                f'top{topk_eval}_correct': topk_correct,
                f'top{topk_eval}_accuracy': topk_correct / total_num,
                'total_num': total_num,
                'kd_weight': self.kd_weight,
                'kd_temperature': self.kd_temperature,
            }
            write_json(osp.join(work_dir, 'metrics.json'), metrics)
            msg = (
                f"==========Test result on {schedule['metric']}==========\n" +
                time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) +
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, "
                f"Top-{topk_eval} correct / Total: {topk_correct}/{total_num}, Top-{topk_eval} accuracy: {topk_correct/total_num}, "
                f"time: {time.time()-last_time}\n"
            )
            log(msg)
