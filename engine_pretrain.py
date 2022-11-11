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
from typing import Iterable

import torch

import wandb

import util.misc as misc
import util.lr_sched as lr_sched

import matplotlib.pyplot as plt


def norm(data:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalize data to have mean=0 and standard_deviation=1

    Parameters
    ----------
    data:  tensor
    """
    mean = torch.mean(data, dim=-1, keepdim=True)
    var = torch.var(data, dim=-1, keepdim=True)

    return (data - mean) / (var + 1e-12)**0.5

def ncc(data_0:torch.Tensor(), data_1:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalized cross-correlation coefficient between two data sets

    Zero-Normalized cross-correlation equals the cosine of the angle between the unit vectors F and T, 
    being thus 1 if and only if F equals T multiplied by a positive scalar. 

    Parameters
    ----------
    data_0, data_1 :  tensors of same size
    """

    nb_of_signals = 1
    for dim in range(data_0.dim()-1): # all but the last dimension (which is the actual signal)
        nb_of_signals = nb_of_signals * data_0.shape[dim]

    cross_corrs = (1.0 / (data_0.shape[-1]-1)) * torch.sum(norm(data=data_0) * norm(data=data_1), dim=-1)

    return (cross_corrs.sum() / nb_of_signals)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
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

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, samples_hat, samples_hat_masked = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        batch_size = samples.shape[0]
        normalized_corr = ncc(samples, samples_hat).item()
        metric_logger.meters['ncc'].update(normalized_corr, n=batch_size)

        #loss_value_reduce = misc.all_reduce_mean(loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)
        log_writer.add_scalar('train/normalized_corr_coef', train_stats["ncc"], epoch)

        if args.wandb == True:
            training_history['epoch'] = epoch
            training_history['train_loss'] = train_stats["loss"]
            training_history['lr'] = train_stats["lr"]
            training_history['Normalized Correlation Coefficient'] = train_stats["ncc"]

            if (epoch % 10) == 0:
                x = samples[0, ..., ::5].detach().cpu().numpy()
                x_hat = samples_hat[0, ..., ::5].detach().cpu().numpy()
                x_hat_masked = samples_hat_masked[0, ..., ::5].detach().cpu().numpy()

                plt.close('all')
                plt.subplot(611)
                plt.plot(range(0, x.shape[-1], 1), x[0, 0, :])
                plt.subplot(612)
                plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :])
                plt.subplot(613)
                plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :])
                plt.subplot(614)
                plt.plot(range(0, x.shape[-1], 1), x[2, 32, :])
                plt.subplot(615)
                plt.plot(range(0, x.shape[-1], 1), x_hat[2, 32, :])
                plt.subplot(616)
                plt.plot(range(0, x.shape[-1], 1), x_hat_masked[2, 32, :])
                plt.tight_layout()
                training_history["Reconstruction"] = wandb.Image(plt)

            # wandb.log(training_history)

    return train_stats, training_history


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, training_history=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        samples = samples.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            loss, samples_hat, samples_hat_masked = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        # batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)

        batch_size = samples.shape[0]
        normalized_corr = ncc(samples, samples_hat).item()
        metric_logger.meters['ncc'].update(normalized_corr, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # log evaluation results
    if log_writer is not None:
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_normalized_corr_coef', test_stats["ncc"], epoch)

        if args.wandb == True:
            training_history['epoch'] = epoch
            training_history['val_loss'] = test_stats["loss"]
            training_history['Val Normalized Correlation Coefficient'] = test_stats["ncc"]

            # if (epoch % 1) == 0:
            #     x = samples[0, ..., ::5].detach().cpu().numpy()
            #     x_hat = samples_hat[0, ..., ::5].detach().cpu().numpy()
            #     x_hat_masked = samples_hat_masked[0, ..., ::5].detach().cpu().numpy()

            #     plt.close('all')
            #     plt.subplot(611)
            #     plt.plot(range(0, x.shape[-1], 1), x[0, 0, :])
            #     plt.subplot(612)
            #     plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :])
            #     plt.subplot(613)
            #     plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :])
            #     plt.subplot(614)
            #     plt.plot(range(0, x.shape[-1], 1), x[2, 32, :])
            #     plt.subplot(615)
            #     plt.plot(range(0, x.shape[-1], 1), x_hat[2, 32, :])
            #     plt.subplot(616)
            #     plt.plot(range(0, x.shape[-1], 1), x_hat_masked[2, 32, :])
            #     plt.tight_layout()
            #     training_history["Val reconstruction"] = wandb.Image(plt)

            wandb.log(training_history)

    return test_stats