# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import argparse

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

import models_mae
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error

from engine_pretrain import train_one_epoch, evaluate_online, evaluate

from util.dataset import SignalDataset
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from memory_profiler import profile
from functools import partial


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch200', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Dataset parameters
    parser.add_argument('--data_path', default='_.pt', type=str,
                    help='dataset path')
    parser.add_argument('--val_data_path', default='', type=str,
                    help='validation dataset path')
    parser.add_argument('--label_map_path', default='', type=str,
                    help='label mappings path')
    parser.add_argument('--val_label_map_path', default='', type=str,
                    help='val label mappings path')
    parser.add_argument('--scale_percentage', default=100, type=int,
                        help='percentage of dataset to sample - for dataset scaling')

    parser.add_argument('--input_channels', type=int, default=5, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=65, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=37000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(5, 65, 37000), type=Tuple,
                        help='images input size')
    parser.add_argument('--electrode_idx', type=str, default=None,
                        help='Comma-separated electrode indices to slice')

    parser.add_argument('--patch_height', type=int, default=65, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=200, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(65, 200), type=Tuple,
                        help='patch size')

    # Label statistics
    parser.add_argument('--train_labels_mean', type=float, default=0.0,
                        help='Mean value of training labels set')
    parser.add_argument('--train_labels_std', type=float, default=1.0,
                        help='stdeof training labels set')

    ##Overfit Case params
    parser.add_argument('--overfit', type=bool, default=False,
                        help='Overfitting case')
    parser.add_argument('--overfit_sample_size', default=10, type=int,
                        help='number of samples to overfit')

    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.add_argument('--ncc_weight', type=float, default=0.1,
                        help='Add normalized cross-correlation (ncc) as additional loss term')
    parser.add_argument('--cos_weight', type=float, default=0.1,
                        help='Add cos similarity as additional loss term')

    # Augmentation parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')
    parser.add_argument('--freq_shift_delta', default=0.005, type=float,
                        help='Delta for the frequency shift (default: 0.005)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    parser.add_argument('--online_evaluation', action='store_true', default=False,
                        help='Perform online evaluation of a downstream task')
    parser.add_argument('--online_evaluation_task', default='classification', type=str,
                        help='Online downstream task (default: classification)')
    parser.add_argument('--online_num_classes', default=2, type=int,
                        help='Online classification task classes (default: 2)')

    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--data_path_online', default='_.pt', type=str,
                        help='dataset path for the online evaluation')
    parser.add_argument('--labels_path_online', default='_.pt', type=str,
                        help='labels path for the online evaluation')
    parser.add_argument('--labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--val_data_path_online', default='', type=str,
                        help='validation dataset path for the online evaluation')
    parser.add_argument('--val_labels_path_online', default='', type=str,
                        help='validation labels path for the online evaluation')
    parser.add_argument('--val_labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--subject_file_path', default='', type=str,
                        help='path to subject id info file')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log (default: ./logs)')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--wandb_name', default='', type=str,
                        help= 'name of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def calculate_baseline_metrics(labels, train_labels_mean):
    """
    Calculate baseline metrics for random guesses based on the mean of training labels.

    Args:
        train_labels (torch.Tensor): Tensor of labels.
        train_labels_mean (float): Mean of the training labels used for making random guesses.

    Returns:
        dict: Dictionary containing MAE and std of random guesses.
    """
    # Create random guesses for both training and validation datasets based on the training labels mean
    random_guess = torch.ones_like(labels) * train_labels_mean
    # Calculate Mean Absolute Error (MAE) for both datasets
    MAE_random_guess = mean_absolute_error(random_guess, labels, multioutput="raw_values")
    # Calculate standard deviation of absolute errors
    std_random_guess = torch.std(torch.abs(random_guess - labels), dim=0, unbiased=True).mean().item()

    return {
        'Random guess MAE': MAE_random_guess,
        'Random guess std': std_random_guess,
    }


def main(args):
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    misc.init_distributed_mode(args)
    #args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.electrode_idx = [int(idx) for idx in args.electrode_idx.split(',')] if args.electrode_idx else None

    # load data
    dataset_train = SignalDataset(data_path=args.data_path, train=True, shape=(266871,19,10000), args=args)
    dataset_val = SignalDataset(data_path=args.val_data_path, train=False, shape=(64818,19,10000), args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # wandb logging
    if args.wandb == True:
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, name= args.wandb_name, id=args.wandb_id, config=config, entity="begum-soenmez")
        else:
            #wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, entity="begum-soenmez")
            wandb.init(project=args.wandb_project, config=config, entity="begum-soenmez")


    # online evaluation
    if args.online_evaluation:
        dataset_online_train = SignalDataset(data_path=args.data_path_online, labels_path=args.labels_path_online,
                                             labels_mask_path=args.labels_mask_path_online,
                                             label_map_path=args.label_map_path,
                                             downstream_task=args.online_evaluation_task, train=True, args=args)

        dataset_online_val = SignalDataset(data_path=args.val_data_path_online, labels_path=args.val_labels_path_online,
                                           labels_mask_path=args.val_labels_mask_path_online,
                                           label_map_path=args.val_label_map_path,
                                           downstream_task=args.online_evaluation_task, train=False, args=args)

        train_labels = dataset_online_train.labels
        val_labels = dataset_online_val.labels

        args.train_labels_mean = train_labels.mean()
        args.train_labels_std = train_labels.std()

    else:
        args.train_labels_mean = 0
        args.train_labels_std = 1


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4
    )

    if args.online_evaluation:
        #Random Guess
        baseline_metrics_train = calculate_baseline_metrics(train_labels, args.train_labels_mean)
        baseline_metrics_val = calculate_baseline_metrics(val_labels, args.train_labels_mean)

        args.baseline_metrics_train = baseline_metrics_train
        args.baseline_metrics_val = baseline_metrics_val

        data_loader_online_train = torch.utils.data.DataLoader(
            dataset_online_train,
            shuffle=True,
            batch_size=256,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

        data_loader_online_val = torch.utils.data.DataLoader(
            dataset_online_val,
            shuffle=False,
            batch_size=256,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

    # define the model
    model = models_mae.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss,
        ncc_weight=args.ncc_weight
    )

    model.to(device, non_blocking=True)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" not in n)
    n_params_decoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" in n)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    print('Number of encoder params (M): %.2f' % (n_params_encoder / 1.e6))
    print('Number of decoder params (M): %.2f' % (n_params_decoder / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")

    eval_criterion = "ncc"

    best_stats = {'loss':np.inf, 'ncc':0.0}
    best_eval_scores = {'count':0, 'nb_ckpts_max':5, 'eval_criterion':[best_stats[eval_criterion]]}

    '''
    #Setup Pytorch Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler('/vol/aimspace/users/soeb/logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
    '''

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                                    log_writer=log_writer, profiler=None, args=args)

        val_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
        print(f"Loss / Normalized CC of the network on the {len(dataset_val)} val images: {val_stats['loss']:.4f}\
            / {val_stats['ncc']:.2f}")

        # online evaluation of the downstream task
        online_history = {}
        if args.online_evaluation and epoch % 5 == 0:
            if args.online_evaluation_task == "classification":
                online_estimator = LogisticRegression(class_weight='balanced', max_iter=2000)
            elif args.online_evaluation_task == "regression":
                online_estimator = LinearRegression()
            online_history = evaluate_online(estimator=online_estimator, model=model, device=device,
                                            train_dataloader=data_loader_online_train,
                                            val_dataloader=data_loader_online_val, args=args)

        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
        best_stats['ncc'] = max(best_stats['ncc'], val_stats['ncc'])

        if eval_criterion == "loss":
            if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion,
                    mode="decreasing")
        else:
            if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion,
                    mode="increasing")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        if args.wandb:
            wandb.log(train_history | test_history | online_history | {"Time per epoch [sec]": total_time})


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        directory_name = f"/vol/aimspace/users/soeb/OPTIMIZATION_PRETRAIN/TUH/PCT{args.scale_percentage}_Pretrain_BATCH{args.batch_size}_BLR{args.blr}_TIME{args.time_steps}_PATCH{args.patch_width}"
        Path(directory_name).mkdir(parents=True, exist_ok=True)

        args.output_dir = directory_name
    main(args)