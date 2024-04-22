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
import glob

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path
import csv

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
# from torch.utils.tensorboard import SummaryWriter
import wandb

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy #, LabelSmoothingCrossEntropy

from util.dataset import SignalDataset
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

import models_vit
from functools import partial

from engine_finetune import train_one_epoch, evaluate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch200', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_channels', type=int, default=5, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=65, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=37000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(5, 65, 37000), type=Tuple,
                        help='images input size')
    parser.add_argument('--electrode_idx', type=int, nargs='*',
                    default=None, help='Electrode indices to slice')

    parser.add_argument('--patch_height', type=int, default=65, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=200, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(65, 200), type=Tuple,
                        help='patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    #Cross-validation params
    parser.add_argument('--k_folds', default=1, type=int,
                    help='Number of folds for k-fold cross-validation, default is 1 = No cross-validation')

    # Augmentation parameters
    parser.add_argument('--masking_blockwise', action='store_true', default=False,
                        help='Masking blockwise in channel and time dimension (instead of random masking)')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_c_ratio', default=0.0, type=float,
                        help='Masking ratio in channel dimension (percentage of removed patches)')
    parser.add_argument('--mask_t_ratio', default=0.0, type=float,
                        help='Masking ratio in time dimension (percentage of removed patches)')

    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')
    parser.add_argument('--freq_shift_delta', default=0.005, type=float,
                        help='Delta for the frequency shift (default: 0.005)')

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 4')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Criterion parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--attention_pool', action='store_true', default=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--downstream_task', default='classification', type=str,
                        help='downstream task (default: classification)')

    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path (default: None)')
    parser.add_argument('--labels_path', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--labels_mask_path', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--label_map_path', default='', type=str,
                    help='label mappings path')
    parser.add_argument('--scale_percentage', default=100, type=int,
                        help='percentage of dataset to sample - for dataset scaling')


    ##Overfit Case params
    parser.add_argument('--overfit', type=bool, default=False,
                        help='Overfitting case')
    parser.add_argument('--overfit_sample_size', default=10, type=int,
                        help='number of samples to overfit')

    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--pos_label', default=0, type=int,
                        help='classification type with the smallest count')

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

    parser.add_argument('--plot_attention_map', action='store_true', default=False)
    parser.add_argument('--plot_embeddings', action='store_true', default=False)
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save model embeddings')
    parser.add_argument('--save_logits', action='store_true', default=False,
                        help='save model logits')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
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

def find_checkpoint_for_fold(base_path, fold_number):
    """
    Function to search for the pretrain checkpoint matching the current fold
    Args:
        base_path: Folder where checkpoint folders are located
        fold_number: current fold

    Returns:
        checkpoint_file: Path to pretrain checkpoint
    """
    # Pattern to match the fold directory based on fold number
    fold_pattern = f"Fold{fold_number}_*"
    full_path_pattern = os.path.join(base_path, fold_pattern)
    fold_dirs = glob.glob(full_path_pattern)

    if not fold_dirs:
        raise FileNotFoundError(f"No directory found for fold {fold_number} in {base_path}")

    # Assuming there's only one matching folder per fold, adjust if necessary
    checkpoint_dir = fold_dirs[0]
    checkpoint_file = glob.glob(os.path.join(checkpoint_dir, '*.pth'))

    if not checkpoint_file:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")

    return checkpoint_file[0]

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

def custom_collate_fn(batch, mean, std):
    """
    Custom collate function for dataloading that accepts varying input sizes.
    Args:
        batch (list of tuples): The batch data - list accepting varying sample sizes
        mean (float): The training set mean
        std (float): The training set std

    Returns:
        list: A list containing normalized batch data with samples, normalized labels, and label masks.
    """
    samples = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_masks = [item[2] for item in batch]
    labels_tensor = torch.stack(labels)

    # Normalize labels
    labels_normalized = (labels_tensor - mean) / (std + 1e-8)
    return [samples, labels_normalized, torch.stack(label_masks)]

def prepare_datasets(dataset, k_folds, args):
    if k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        splits = list(kf.split(dataset))
    else:
        # If no cross-validation, just split the dataset into train and validation in some ratio (e.g., 80-20)
        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        indices = torch.randperm(num_samples).tolist()
        splits = [(indices[:train_size], indices[train_size:])]

    return splits

def main(args):
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    # misc.init_distributed_mode(args)
    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Create empty .csv file for saving
    csv_filename = f"benchmark-mae_BATCH{args.batch_size}_BLR{args.blr}_dataset-lemon_chkpt{args.finetune}.csv"
    csv_directory = "/vol/aimspace/users/soeb/lemon_kfold_results"
    csv_file_path = os.path.join(csv_directory, csv_filename)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["fold", "MAE", "r2", "benchmark", "dataset"])
        writer.writeheader()

    dataset = SignalDataset(data_path=args.data_path, labels_path=args.labels_path,
                                  labels_mask_path=args.labels_mask_path,
                                  label_map_path=args.label_map_path,
                                  downstream_task=args.downstream_task, train=True, args=args)
    splits = prepare_datasets(dataset, args.k_folds, args)

    # train balanced
    class_weights = 2.0 / (2.0 * torch.Tensor([1.0, 1.0])) # total_nb_samples / (nb_classes * samples_per_class)

    '''
    if False:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            c
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    '''

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    elif False: #args.log_dir and args.eval and "checkpoint" not in args.resume.split("/")[-1]:
        log_writer = SummaryWriter(log_dir=args.log_dir + "/eval")
    else:
        log_writer = None

###################################  K-FOLD CROSS VALIDATION  ###############################################
    # Combining the initial datasets if k_folds > 1
    for fold, (train_idx, val_idx) in enumerate(splits):
        #Find checkpoint file to load
        try:
            args.finetune = find_checkpoint_for_fold("/vol/aimspace/users/soeb/EXP1_output_pretrain_lemon_kfold", fold)
        except FileNotFoundError as e:
            print(e)
            continue

        if args.wandb:
            config = vars(args)
            if wandb.run is not None:
                wandb.finish()
            # Start a new wandb run for the current fold
            wandb.init(project=args.wandb_project, config=config,
                    resume=False, id=None, group=f"Finetune_10Fold_BATCH{args.batch_size}_BLR{args.blr}",
                    name=f"Fold_{fold}", entity="begum-soenmez")

        print(f"Processing fold {fold}/{max(1, args.k_folds)}")

         # Subset the full dataset into train and validation datasets for the current fold
        dataset_train_fold = SignalDataset(data_path=args.data_path, labels_path=args.labels_path,
                                  labels_mask_path=args.labels_mask_path,
                                  label_map_path=args.label_map_path,
                                  downstream_task=args.downstream_task, train=True, indices=train_idx, args=args)

        dataset_val_fold = SignalDataset(data_path=args.data_path, labels_path=args.labels_path,
                                  labels_mask_path=args.labels_mask_path,
                                  label_map_path=args.label_map_path,
                                  downstream_task=args.downstream_task, train=False, indices=val_idx, args=args)

        print(f"Training set size: {len(dataset_train_fold)}")
        print(f"Validation set size: {len(dataset_val_fold)}")

        # Get train label statistics for normalization
        train_labels = dataset_train_fold.labels
        val_labels= dataset_val_fold.labels
        args.train_labels_mean = train_labels.mean(dim=0)
        args.train_labels_std = train_labels.std(dim=0)

        print(f"Fold {fold} Training labels mean: %.5f" % args.train_labels_mean)
        print(f"Fold {fold} Training labels std: %.5f" % args.train_labels_std)

        # This is to dynamically change the mean and std for every fold
        collate_fn_with_stats = partial(custom_collate_fn, mean=args.train_labels_mean, std=args.train_labels_std)

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train_fold, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val_fold)

        data_loader_train = DataLoader(
            dataset_train_fold,
            sampler=sampler_train,
             #shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_with_stats
            )

        data_loader_val = DataLoader(
            dataset_val_fold,
            sampler=sampler_val,
            #shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_with_stats
            )

        #Random guessing stats
        baseline_metrics_train = calculate_baseline_metrics(train_labels, args.train_labels_mean)
        baseline_metrics_val = calculate_baseline_metrics(val_labels, args.train_labels_mean)

        args.baseline_metrics_train = baseline_metrics_train
        args.baseline_metrics_val = baseline_metrics_val

        #REMAINDER OF THE CODE
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            patch_size=args.patch_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            attention_pool=args.attention_pool,
            masking_blockwise=args.masking_blockwise,
            mask_ratio=args.mask_ratio,
            mask_c_ratio=args.mask_c_ratio,
            mask_t_ratio=args.mask_t_ratio
        )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            checkpoint_input_size = checkpoint['args'].input_size
            interpolate_pos_embed(model, checkpoint_model, checkpoint_input_size)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            elif args.attention_pool:
                pass
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=0.01)#2e-5)

        model.to(device, non_blocking=True)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print("Number of params (M): %.2f" % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 4

        print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()

        class_weights = class_weights.to(device=device, non_blocking=True)
        if args.downstream_task == 'regression':
            criterion = torch.nn.MSELoss()
        elif mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        print("criterion = %s" % str(criterion))

        if not args.eval:
            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            sub_strings = args.resume.split("/")
            if "checkpoint" in sub_strings[-1]:
                nb_ckpts = 1
            else:
                nb_ckpts = int(sub_strings[-1])+1

            for epoch in range(0, nb_ckpts):
                if "checkpoint" not in sub_strings[-1]:
                    args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"

                misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

                test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
                if args.downstream_task == 'classification':
                    print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val_fold)} test images: {test_stats['acc']:.2f}% /"
                        f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['auprc']:.2f}%")
                elif args.downstream_task == 'regression':
                    print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / MAE in Years / Pearson Correlation Coefficient (PCC) / R Squared (R2)",
                        f"of the network on the {len(dataset_val_fold)} test images: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} / {test_stats['mae_years']:.4f} /",
                        f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")

                if args.wandb:
                    wandb.log(test_history)

            exit(0)

        # Define callbacks
        early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

        print(f"Start training for {args.epochs} epochs")

        if args.downstream_task == 'classification':
            eval_criterion = "auroc"
        elif args.downstream_task == 'regression':
            eval_criterion = "pcc"

        best_stats = {'loss':np.inf, 'acc':0.0, 'f1':0.0, 'auroc':0.0, 'auprc':0.0, 'rmse':np.inf, 'mae':np.inf, 'mae_years':np.inf, 'pcc':0.0, 'r2':-1.0}
        best_eval_scores = {'count':0, 'nb_ckpts_max':5, 'eval_criterion':[best_stats[eval_criterion]]}
        for epoch in range(args.start_epoch, args.epochs):
            start_time = time.time()

            if True: #args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats, train_history = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                                        loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)

            test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)

            if eval_criterion == "loss" or eval_criterion == "rmse" or eval_criterion == "mae":
                if early_stop.evaluate_decreasing_metric(val_metric=test_stats[eval_criterion]):
                    break
                if args.output_dir and test_stats[eval_criterion] <= max(best_eval_scores['eval_criterion']):
                    # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                    if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                        best_eval_scores['count'] += 1
                    else:
                        best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                        best_eval_scores['eval_criterion'].pop()
                    best_eval_scores['eval_criterion'].append(test_stats[eval_criterion])

                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion,
                        mode="decreasing")
            else:
                if early_stop.evaluate_increasing_metric(val_metric=test_stats[eval_criterion]):
                    break
                if args.output_dir and test_stats[eval_criterion] >= min(best_eval_scores['eval_criterion']):
                    # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                    if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                        best_eval_scores['count'] += 1
                    else:
                        best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                        best_eval_scores['eval_criterion'].pop()
                    best_eval_scores['eval_criterion'].append(test_stats[eval_criterion])

                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion,
                        mode="increasing")

            best_stats['loss'] = min(best_stats['loss'], test_stats['loss'])

            if args.downstream_task == 'classification':
                # update best stats
                best_stats['f1'] = max(best_stats['f1'], test_stats['f1'])
                best_stats['acc'] = max(best_stats['acc'], test_stats["acc"])
                best_stats['auroc'] = max(best_stats['auroc'], test_stats['auroc'])
                best_stats['auprc'] = max(best_stats['auprc'], test_stats['auprc'])

                print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}% /",
                    f"{test_stats['f1']:.1f}% / {test_stats['auroc']:.1f}% / {test_stats['auprc']:.1f}%")
                print(f'Max Accuracy / F1 / AUROC / AUPRC: {best_stats["acc"]:.2f}% / {best_stats["f1"]:.2f}% /',
                    f'{best_stats["auroc"]:.2f}% / {best_stats["auprc"]:.2f}%\n')

            elif args.downstream_task == 'regression':
                # update best stats
                best_stats['rmse'] = min(best_stats['rmse'], test_stats['rmse'])
                best_stats['mae'] = min(best_stats['mae'], test_stats['mae'])
                best_stats['mae_years'] = min(best_stats['mae_years'], test_stats['mae_years'])
                best_stats['pcc'] = max(best_stats['pcc'], test_stats['pcc'])
                best_stats['r2'] = max(best_stats['r2'], test_stats['r2'])

                print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / MAE in Years / Pearson Correlation Coefficient (PCC) / R Squared (R2)",
                    f"of the network on the {len(dataset_val_fold)} test images: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} / {test_stats['mae_years']:.4f} /",
                    f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")
                print(f'Min Root Mean Squared Error (RMSE) / Min Mean Absolute Error (MAE) / Min MAE in Years / Max Pearson Correlation Coefficient (PCC) /',
                    f'Max R Squared (R2): {best_stats["rmse"]:.4f} / {best_stats["mae"]:.4f} / {best_stats["mae_years"]:.4f} / {best_stats["pcc"]:.4f} / {best_stats["r2"]:.4f}\n')

            log_stats = {**{f'train_{k}': str(v) for k, v in train_stats.items()},
                            **{f'test_{k}': str(v) for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            total_time = time.time() - start_time
            if args.wandb:
                wandb.log(train_history | test_history | {"Time per epoch [sec]": total_time})

        if args.wandb:
            wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})
            wandb.finish()

        # Log the best statistics for the current fold
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["fold", "MAE", "r2", "benchmark", "dataset"])
            writer.writerow({
                "fold": fold,
                "MAE": best_stats["mae_years"],
                "r2": best_stats["r2"],
                "benchmark": "mae",
                "dataset": "lemon"
            })



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        if args.overfit == False:
            directory_name = f"/vol/aimspace/users/soeb/output_finetune/output_pretrain_lemon/NEW_Finetune_P{args.patch_width}_T{args.time_steps}_BATCH{args.batch_size}_BLR{args.blr}"
            Path(directory_name).mkdir(parents=True, exist_ok=True)

            args.output_dir = directory_name
    main(args)