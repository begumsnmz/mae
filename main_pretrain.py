# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path

import sys

import torch
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import wandb

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# sys.path.append("..")
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch, evaluate

from util.dataset import EEGDatasetFast


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch200', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_channels', type=int, default=5, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=65, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=37000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(5, 65, 37000), type=Tuple,
                        help='images input size')

    parser.add_argument('--patch_height', type=int, default=65, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=200, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(65, 200), type=Tuple,
                        help='patch size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

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

    # Dataset parameters
    parser.add_argument('--data_path', default='data_8fold_decomposed_2d_all.pt', type=str,
                        help='dataset path')
    parser.add_argument('--labels_path', default='labels_2classes_8fold_decomposed_2d_fs200.pt', type=str,
                        help='labels path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', action='store_true', default=False)
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

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--transfer_learning', action='store_true', default=False)
    parser.add_argument('--transfer_data_path', default='data_SEED_decomposed_2d_fs200.pt', type=str,
                        help='transfer learning dataset path')
    parser.add_argument('--transfer_labels_path', default='labels_2classes_SEED_fs200.pt', type=str,
                        help='transfer learning labels path')

    return parser


def main(args):
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    # load data
    dataset_mri = EEGDatasetFast(augment=True, args=args)
    dataset_mri_train = Subset(dataset_mri, list(range(int(0*1), int(114*1))))
    # dataset_mri_train = Subset(dataset_mri, list(range(int(0*1), int(138*1))))

    if args.transfer_learning == True:
        # GENERAL
        args.data_path = args.transfer_data_path
        args.labels_path = args.transfer_labels_path
        dataset_external = EEGDatasetFast(augment=True, args=args)
        dataset_train = ConcatDataset([dataset_mri_train, dataset_external])

        # # SEED
        # args.data_path = "/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_decomposed_ideal_fs200.pt"
        # args.labels_path = "/home/oturgut/PyTorchEEG/data/preprocessed/labels_3classes_SEED_fs200.pt"
        # dataset_seed = EEGDatasetFast(augment=True, args=args)
        # # dataset_seed = Subset(dataset_seed, list(range(0, 448)))
        # # dataset_seed = ConcatDataset([Subset(dataset_seed, list(range(0, 112))), Subset(dataset_seed, list(range(224, 559)))])
        # # dataset_train = dataset_seed
        # dataset_train = ConcatDataset([dataset_mri_train, dataset_seed])

        # # MOIM
        # args.data_path = "/home/oturgut/PyTorchEEG/data/preprocessed/data_MOIM_snippets60s_decomposed_ideal_fs200.pt"
        # args.labels_path = "/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_MOIM_snippets60s_fs200.pt"
        # dataset_moim = EEGDatasetFast(augment=True, args=args)
        # dataset_train = ConcatDataset([dataset_mri_train, dataset_moim])

        # # LEMON
        # args.data_path = "/home/oturgut/PyTorchEEG/data/preprocessed/data_LEMON_ec_decomposed_2d_fs200.pt"
        # args.labels_path = "/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_LEMON_fs200.pt"
        # dataset_lemon_ec = EEGDatasetFast(augment=True, args=args)

        # args.data_path = "/home/oturgut/PyTorchEEG/data/preprocessed/data_LEMON_eo_decomposed_2d_fs200.pt"
        # args.labels_path = "/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_LEMON_fs200.pt"
        # dataset_lemon_eo = EEGDatasetFast(augment=True, args=args)
        # dataset_train = ConcatDataset([dataset_mri_train, dataset_lemon_ec, dataset_lemon_eo])
    else:
        dataset_train = dataset_mri_train
    
    dataset_mri_validate = EEGDatasetFast(transform=True, augment=False, args=args)
    dataset_val = Subset(dataset_mri_validate, list(range(int(114*1), int(152*1))))
    # dataset_val = Subset(dataset_mri, list(range(int(138*1), int(184*1))))

    print("Dataset size: ", len(dataset_train))

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

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

        if args.wandb == True:
            config = vars(args)
            wandb.init(project="MAE_He", config=config, entity="oturgut")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss
    )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, training_history = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, training_history=training_history, args=args)
        print(f"Loss / Normalized CC of the network on the {len(dataset_val)} val images: {val_stats['loss']:.4f} / {val_stats['ncc']:.2f}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
