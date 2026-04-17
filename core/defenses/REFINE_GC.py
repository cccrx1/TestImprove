import os
import os.path as osp
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Base
from ..models import GeoCorrectionNet
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


class REFINE_GC(Base):
    """REFINE with a lightweight geometric correction front-end."""

    def __init__(
        self,
        unet,
        model,
        pretrain=None,
        arr_path=None,
        num_classes=10,
        lmd=0.1,
        supcon_temperature=0.07,
        enable_label_shuffle=True,
        norm_mean=None,
        norm_std=None,
        geo_channels=16,
        gc_max_delta=0.08,
        grid_reg_weight=0.01,
        grid_smooth_weight=0.01,
        geo_pretrain=None,
        seed=0,
        deterministic=False,
    ):
        super().__init__(seed=seed, deterministic=deterministic)
        self.unet = unet
        self.geo_net = GeoCorrectionNet(
            in_channels=3,
            base_channels=geo_channels,
            max_delta=gc_max_delta,
        )
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.num_classes = num_classes
        self.lmd = lmd
        self.supcon_temperature = supcon_temperature
        self.enable_label_shuffle = bool(enable_label_shuffle)
        self.grid_reg_weight = float(grid_reg_weight)
        self.grid_smooth_weight = float(grid_smooth_weight)
        if norm_mean is not None and norm_std is not None:
            self.norm_mean = torch.tensor(norm_mean, dtype=torch.float32).view(1, 3, 1, 1)
            self.norm_std = torch.tensor(norm_std, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            self.norm_mean = None
            self.norm_std = None
        if pretrain is not None:
            self.unet.load_state_dict(torch.load(pretrain), strict=False)
        if geo_pretrain is not None:
            self.geo_net.load_state_dict(torch.load(geo_pretrain), strict=False)
        if arr_path is not None:
            self.arr_shuffle = np.array(torch.load(arr_path))
            if not self.enable_label_shuffle:
                self.arr_shuffle = np.arange(self.num_classes, dtype=np.int64)
        else:
            if self.num_classes is None:
                raise ValueError('num_classes must be provided when arr_path is not specified.')
            self.init_label_shuffle()

    def init_label_shuffle(self):
        if not self.enable_label_shuffle:
            self.arr_shuffle = np.arange(self.num_classes, dtype=np.int64)
            return
        arr = np.arange(self.num_classes)
        arr_shuffle = np.arange(self.num_classes)
        while True:
            num = sum(arr == arr_shuffle)
            if num == 0:
                break
            np.random.shuffle(arr_shuffle)
        self.arr_shuffle = arr_shuffle

    def label_shuffle(self, label):
        if not self.enable_label_shuffle:
            return label
        index = torch.from_numpy(self.arr_shuffle).to(label.device).repeat(label.shape[0], 1)
        return torch.zeros_like(label).scatter(1, index, label)

    def _denormalize(self, image):
        if self.norm_mean is None or self.norm_std is None:
            return image
        norm_std = self.norm_std.to(image.device)
        norm_mean = self.norm_mean.to(image.device)
        return image * norm_std + norm_mean

    def _normalize(self, image):
        if self.norm_mean is None or self.norm_std is None:
            return image
        norm_std = self.norm_std.to(image.device)
        norm_mean = self.norm_mean.to(image.device)
        return (image - norm_mean) / norm_std

    def _identity_grid(self, height, width, device):
        array1d_y = torch.linspace(-1, 1, steps=height, device=device)
        array1d_x = torch.linspace(-1, 1, steps=width, device=device)
        y, x = torch.meshgrid(array1d_y, array1d_x, indexing='ij')
        return torch.stack((x, y), dim=-1).unsqueeze(0)

    def _rectify(self, clean_image):
        delta = self.geo_net(clean_image)
        delta_grid = delta.permute(0, 2, 3, 1)
        identity = self._identity_grid(clean_image.shape[-2], clean_image.shape[-1], clean_image.device)
        sampling_grid = torch.clamp(identity + delta_grid, -1, 1)
        rectified = F.grid_sample(clean_image, sampling_grid, align_corners=True)
        return rectified, delta

    def _grid_regularization(self, delta):
        reg = delta.pow(2).mean()
        if self.grid_smooth_weight <= 0:
            return reg
        dy = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        dx = delta[:, :, :, 1:] - delta[:, :, :, :-1]
        smooth = dx.abs().mean() + dy.abs().mean()
        return reg + self.grid_smooth_weight * smooth

    def _reprogram_and_classify(self, image):
        clean_image = self._denormalize(image)
        rectified, delta = self._rectify(clean_image)
        x_adv = torch.clamp(self.unet(rectified), 0, 1)
        y_adv = self.model(self._normalize(x_adv))
        return rectified, x_adv, y_adv, delta

    def _augment_view(self, image):
        if image.size(-1) <= 1:
            return image
        flip_mask = (torch.rand(image.size(0), 1, 1, 1, device=image.device) > 0.5)
        flipped = torch.flip(image, dims=[3])
        return torch.where(flip_mask, flipped, image)

    def _autocast(self, enabled):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast('cuda', enabled=enabled)
        return torch.cuda.amp.autocast(enabled=enabled)

    def _make_grad_scaler(self, enabled):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            return torch.amp.GradScaler('cuda', enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def forward(self, image):
        self.X_rectified, self.X_adv, self.Y_adv, self.delta_grid = self._reprogram_and_classify(image)
        y_adv = F.softmax(self.Y_adv, 1)
        return self.label_shuffle(y_adv)

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _test(self, dataset, device, batch_size=16, num_workers=8, loss_func=None, supconloss_func=None, amp=False):
        with torch.no_grad():
            loss_func = loss_func or torch.nn.BCELoss(reduction='none')
            if supconloss_func is None:
                supconloss_func = SupConLoss(
                    temperature=self.supcon_temperature,
                    base_temperature=self.supcon_temperature,
                )
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            self.geo_net = self.geo_net.to(device)
            self.geo_net.eval()
            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)

            losses = []
            for batch in test_loader:
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                with self._autocast(enabled=amp):
                    f_logit = self.model(batch_img)
                    f_index = f_logit.argmax(1)
                    f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                    logit = self.forward(batch_img)
                    batch_img_aug = self._augment_view(batch_img)
                    _, _, y_adv_aug, _ = self._reprogram_and_classify(batch_img_aug)
                    features = torch.cat(
                        [
                            F.normalize(self.Y_adv, dim=1).unsqueeze(1),
                            F.normalize(y_adv_aug, dim=1).unsqueeze(1),
                        ],
                        dim=1,
                    )
                    supconloss = supconloss_func(features, f_index)

                cls_loss = loss_func(logit.float(), f_label.float())
                grid_loss = self._grid_regularization(self.delta_grid.float())
                loss = cls_loss + self.lmd * supconloss.float() + self.grid_reg_weight * grid_loss
                losses.append(loss.cpu())

            losses = torch.cat(losses, dim=0)
            return losses.mean()

    def train_unet(self, train_dataset, test_dataset, schedule):
        schedule = deepcopy(schedule)
        schedule.setdefault('defense_name', 'refine_gc')

        if self.num_classes is None:
            self.num_classes = infer_num_classes(train_dataset)
            self.init_label_shuffle()

        if 'pretrain' in schedule:
            self.unet.load_state_dict(torch.load(schedule['pretrain']), strict=False)
        if 'geo_pretrain' in schedule:
            self.geo_net.load_state_dict(torch.load(schedule['geo_pretrain']), strict=False)
        if 'arr_path' in schedule:
            self.arr_shuffle = np.array(torch.load(schedule['arr_path']))
            if not self.enable_label_shuffle:
                self.arr_shuffle = np.arange(self.num_classes, dtype=np.int64)

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker,
        )

        self.geo_net = self.geo_net.to(device)
        self.geo_net.train()
        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='mean')
        supconloss_func = SupConLoss(
            temperature=self.supcon_temperature,
            base_temperature=self.supcon_temperature,
        )
        optimizer = torch.optim.Adam(
            list(self.geo_net.parameters()) + list(self.unet.parameters()),
            schedule['lr'],
            schedule['betas'],
            schedule['eps'],
            schedule['weight_decay'],
            schedule['amsgrad'],
        )
        use_amp = (device.type == 'cuda') and bool(schedule.get('amp', True))
        scaler = self._make_grad_scaler(enabled=use_amp)

        work_dir = resolve_output_dir(schedule, stage='defenses', method_name='refine_gc')
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))
        write_json(
            osp.join(work_dir, 'config.json'),
            build_run_metadata(
                schedule,
                stage='defenses',
                defense_name='refine_gc',
                output_dir=work_dir,
                num_classes=self.num_classes,
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
        )
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']

            for batch_id, batch in enumerate(train_loader):
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                with self._autocast(enabled=use_amp):
                    f_logit = self.model(batch_img)
                    f_index = f_logit.argmax(1)
                    f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                    logit = self.forward(batch_img)

                    batch_img_aug = self._augment_view(batch_img)
                    _, _, y_adv_aug, _ = self._reprogram_and_classify(batch_img_aug)
                    features = torch.cat(
                        [
                            F.normalize(self.Y_adv, dim=1).unsqueeze(1),
                            F.normalize(y_adv_aug, dim=1).unsqueeze(1),
                        ],
                        dim=1,
                    )
                    supconloss = supconloss_func(features, f_index)

                cls_loss = loss_func(logit.float(), f_label.float())
                grid_loss = self._grid_regularization(self.delta_grid.float())
                loss = cls_loss + self.lmd * supconloss.float() + self.grid_reg_weight * grid_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                iteration += 1
                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = (
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +
                        f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset)//schedule['batch_size']}, "
                        f"lr: {schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    )
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(
                    test_dataset,
                    device,
                    schedule['batch_size'],
                    schedule['num_workers'],
                    supconloss_func=supconloss_func,
                    amp=use_amp,
                )
                metrics = {
                    'loss': float(loss),
                    'epoch': i + 1,
                    'stage': 'defenses',
                    'defense_name': 'refine_gc',
                    'metric': 'train_unet_test_loss',
                }
                write_json(osp.join(work_dir, 'metrics.json'), metrics)
                msg = (
                    "==========Test result on test dataset==========\n" +
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +
                    f"loss: {loss}, time: {time.time()-last_time}\n"
                )
                log(msg)
                self.geo_net = self.geo_net.to(device)
                self.geo_net.train()
                self.unet = self.unet.to(device)
                self.unet.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.geo_net.eval()
                self.unet.eval()
                self.geo_net = self.geo_net.cpu()
                self.unet = self.unet.cpu()
                torch.save(self.unet.state_dict(), os.path.join(work_dir, f'ckpt_epoch_{i+1}.pth'))
                torch.save(self.geo_net.state_dict(), os.path.join(work_dir, f'geo_ckpt_epoch_{i+1}.pth'))
                self.geo_net = self.geo_net.to(device)
                self.unet = self.unet.to(device)
                self.geo_net.train()
                self.unet.train()

        return work_dir

    def test(self, dataset, schedule):
        schedule = deepcopy(schedule)
        schedule.setdefault('defense_name', 'refine_gc')
        work_dir = resolve_output_dir(
            schedule,
            stage='defenses',
            method_name='refine_gc',
            extra_tag=schedule.get('metric'),
        )
        log = Log(osp.join(work_dir, 'log.txt'))
        write_json(
            osp.join(work_dir, 'config.json'),
            build_run_metadata(
                schedule,
                stage='defenses',
                defense_name='refine_gc',
                output_dir=work_dir,
                num_classes=self.num_classes,
            ),
        )

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

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

            self.geo_net = self.geo_net.to(device)
            self.geo_net.eval()
            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)
            self.model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
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
            top5_k = topk[-1]
            prec5 = metric_map[top5_k]
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            metrics = {
                'stage': 'defenses',
                'defense_name': 'refine_gc',
                'metric': schedule['metric'],
                'top1_correct': top1_correct,
                'top1_accuracy': top1_correct / total_num,
                f'top{top5_k}_correct': top5_correct,
                f'top{top5_k}_accuracy': top5_correct / total_num,
                'total_num': total_num,
            }
            write_json(osp.join(work_dir, 'metrics.json'), metrics)
            msg = (
                f"==========Test result on {schedule['metric']}==========\n" +
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, "
                f"Top-{top5_k} correct / Total: {top5_correct}/{total_num}, Top-{top5_k} accuracy: {top5_correct/total_num}, "
                f"time: {time.time()-last_time}\n"
            )
            log(msg)
