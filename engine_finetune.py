# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
from timm import data

import torch
import torchmetrics

import wandb

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        training_history = {}
    
    metric_f1 = torchmetrics.F1Score(num_classes=2, threshold=0.5, average='macro').to(device=device)
    metric_auroc = torchmetrics.AUROC(num_classes=2, pos_label=0, average='macro').to(device=device)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        f1 = metric_f1(outputs, targets)
        auroc = metric_auroc(outputs, targets)

        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1'].update(100*f1.item(), n=batch_size)
        metric_logger.meters['auroc'].update(100*auroc.item(), n=batch_size)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            log_writer.add_scalar('perf/train_acc1', acc1, epoch_1000x)
            #log_writer.add_scalar('perf/train_acc5', acc5, epoch_1000x)
            log_writer.add_scalar('perf/train_f1', f1, epoch_1000x)
            log_writer.add_scalar('perf/train_auroc', auroc, epoch_1000x)

            if args.wandb == True:
                training_history['epoch_1000x'] = epoch_1000x
                training_history['loss'] = loss_value_reduce
                training_history['lr'] = max_lr
                training_history['acc'] = acc1
                training_history['f1'] = f1
                training_history['auroc'] = auroc
                wandb.log(training_history)

    f1 = 100*metric_f1.compute() # returns the f1 score for class 0
    metric_f1.reset()
    auroc = 100*metric_auroc.compute() # returns the auroc for both classes combined (see average="macro")
    metric_auroc.reset()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    metric_f1 = torchmetrics.F1Score(num_classes=2, threshold=0.5, average='macro').to(device=device)
    metric_auroc = torchmetrics.AUROC(num_classes=2, pos_label=0, average='macro').to(device=device)
    # #### THIS IS ONLY FOR SEED ####
    # metric_f1 = torchmetrics.F1Score(num_classes=3, threshold=0.5, average=None).to(device=device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        print("Target: ", [i.item() for i in target])
        print("Output: ", [i.item() for i in torch.argmax(output, dim=1)])

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        f1 = metric_f1(output, target)
        auroc = metric_auroc(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['f1'].update(100*f1.item(), n=batch_size)
        metric_logger.meters['auroc'].update(100*auroc.item(), n=batch_size)

    f1 = 100*metric_f1.compute() # returns the f1 score for class 0
    metric_f1.reset()
    auroc = 100*metric_auroc.compute() # returns the auroc for both classes combined (see average="macro")
    metric_auroc.reset()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} F1 (class 0) {f1.global_avg:.3f} AUROC {auroc.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, f1=metric_logger.f1, auroc=metric_logger.auroc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}