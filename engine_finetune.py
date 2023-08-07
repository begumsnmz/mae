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

import math
import sys
from typing import Iterable, Optional
from timm import data

import torch
import torchmetrics
from torchmetrics import MultioutputWrapper

import sklearn.metrics

import wandb

import umap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import util.plot as plot
from util.metrics import MeanSquaredError


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
    
    # classification metrics
    metric_acc = torchmetrics.Accuracy(num_classes=args.nb_classes, threshold=0.5, average='weighted').to(device=device)
    metric_f1 = torchmetrics.F1Score(num_classes=args.nb_classes, threshold=0.5, average=None).to(device=device)
    metric_auroc = torchmetrics.AUROC(num_classes=args.nb_classes, pos_label=args.pos_label, average='macro').to(device=device)
    preds = []
    trgts = []

    # regression metrics
    metric_rmse = MeanSquaredError(squared=False) #.to(device=device)
    metric_pcc = MultioutputWrapper(torchmetrics.PearsonCorrCoef(num_outputs=1), num_outputs=args.nb_classes).to(device=args.device)

    for data_iter_step, (samples, targets, targets_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets * targets_mask

        if args.downstream_task == 'classification':
            targets_mask = targets_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)*targets_mask
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

        if args.downstream_task == 'classification':
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc = metric_acc(outputs.argmax(dim=-1), targets)
            f1 = metric_f1(outputs.argmax(dim=-1), targets)[args.pos_label]
            auroc = metric_auroc(torch.nn.functional.softmax(outputs, dim=-1)[:, args.pos_label], targets)
            auprc = sklearn.metrics.average_precision_score(y_true=targets.detach().cpu(), y_score=torch.nn.functional.softmax(outputs.detach().cpu().type(torch.float32), dim=-1)[:, args.pos_label], average='micro', pos_label=args.pos_label)
            # store the results of each step in a list to calculate the auc globally of the entire epoch
            [preds.append(elem.item()) for elem in torch.nn.functional.softmax(outputs, dim=-1)[:, args.pos_label]]
            [trgts.append(elem.item()) for elem in targets]

            batch_size = samples.shape[0]
            # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc'].update(100*acc.item(), n=batch_size)
            metric_logger.meters['f1'].update(100*f1.item(), n=batch_size)
            metric_logger.meters['auroc'].update(100*auroc.item(), n=batch_size)
            metric_logger.meters['auprc'].update(100*auprc.item(), n=batch_size)
        elif args.downstream_task == 'regression':
            rmse = metric_rmse(outputs, targets)
            pcc = metric_pcc(outputs, targets)

            batch_size = samples.shape[0]
            metric_logger.meters['rmse'].update(torch.tensor(rmse).mean().item(), n=batch_size)
            metric_logger.meters['pcc'].update(torch.tensor(pcc).mean().item(), n=batch_size)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    training_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.downstream_task == 'classification':
        acc = 100*metric_acc.compute()
        metric_acc.reset()
        training_stats["acc"] = acc.item()

        f1 = 100*metric_f1.compute()[args.pos_label] # returns the f1 score
        metric_f1.reset()
        training_stats["f1"] = f1.item()

        # reset the auc metric to calculate the auc globally of the entire epoch
        metric_auroc.reset()
        metric_auroc(torch.tensor(preds, dtype=torch.float), torch.tensor(trgts, dtype=torch.long))
        auroc = 100*metric_auroc.compute() # returns the auroc for both classes combined (see average="macro")
        metric_auroc.reset()
        training_stats["auroc"] = auroc.item()

        auprc = 100*sklearn.metrics.average_precision_score(y_true=trgts, y_score=preds, average='micro', pos_label=args.pos_label)
        training_stats["auprc"] = auprc.item()
    elif args.downstream_task == 'regression':
        rmse = metric_rmse.compute()
        metric_rmse.reset()
        training_stats["rmse"] = torch.tensor(rmse).mean().item()

        pcc = metric_pcc.compute()
        metric_pcc.reset()
        training_stats["pcc"] = torch.tensor(pcc).mean().item()

    if log_writer is not None: #and (data_iter_step + 1) % accum_iter == 0:
        #""" We use epoch_1000x as the x-axis in tensorboard.
        #This calibrates different curves when batch size changes.
        #"""
        #epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        log_writer.add_scalar('loss', training_stats["loss"], epoch)
        log_writer.add_scalar('lr', training_stats["lr"], epoch)

        if args.downstream_task == 'classification':
            # log_writer.add_scalar('perf/train_acc1', training_stats["acc1"], epoch)
            #log_writer.add_scalar('perf/train_acc5', acc5, epoch)
            log_writer.add_scalar('perf/train_acc', acc, epoch)
            log_writer.add_scalar('perf/train_f1', training_stats["f1"], epoch)
            log_writer.add_scalar('perf/train_auroc', training_stats["auroc"], epoch)
            log_writer.add_scalar('perf/train_auprc', training_stats["auprc"], epoch)
        elif args.downstream_task == 'regression':
            log_writer.add_scalar('perf/train_rmse', training_stats["rmse"], epoch)
            log_writer.add_scalar('perf/train_pcc', training_stats["pcc"], epoch)

        if args.wandb == True:
            training_history['epoch'] = epoch
            training_history['loss'] = training_stats["loss"]
            training_history['lr'] = training_stats["lr"]
            if args.downstream_task == 'classification':
                training_history['acc'] = training_stats["acc"]
                training_history['f1'] = training_stats["f1"]
                training_history['auroc'] = training_stats["auroc"]
                training_history['auprc'] = training_stats["auprc"]
            elif args.downstream_task == 'regression':
                training_history['rmse'] = training_stats["rmse"]
                training_history['pcc'] = training_stats["pcc"]

                for i in range(targets.shape[-1]):
                    training_history[f'Train/RMSE/{i}'] = torch.tensor(rmse[i]).item()
                    training_history[f'Train/PCC/{i}'] = pcc[i].item()

            # wandb.log(training_history)

    return training_stats, training_history


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    if args.downstream_task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        test_history = {}  

    # switch to evaluation mode
    # classificaiton metrics
    model.eval()
    metric_acc = torchmetrics.Accuracy(num_classes=args.nb_classes, threshold=0.5, average='weighted').to(device=device)
    metric_f1 = torchmetrics.F1Score(num_classes=args.nb_classes, threshold=0.5, average=None).to(device=device)
    metric_auroc = torchmetrics.AUROC(num_classes=args.nb_classes, pos_label=args.pos_label, average='macro').to(device=device)
    preds = []
    trgts = []
    embeddings = torch.Tensor().to(device=args.device)
    predictions = torch.Tensor().to(device=args.device)
    # tp, fp, tn, fn = 0, 0, 0, 0

    # regression metrics
    metric_rmse = MeanSquaredError(squared=False) #.to(device=device)
    metric_pcc = MultioutputWrapper(torchmetrics.PearsonCorrCoef(num_outputs=1), num_outputs=args.nb_classes).to(device=args.device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-2]
        target_mask = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        target = target * target_mask

        if args.downstream_task == 'classification':
            target_mask = target_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)*target_mask
            embedding = model.forward_features(images)
            output = model.forward_head(embedding)
            output = output*target_mask
            loss = criterion(output, target)

        attention_map = model.blocks[-1].attn.attn_map

        # print("Target: ", [i.item() for i in target])
        # print("Output: ", [i.item() for i in torch.argmax(output, dim=1)])

        embeddings = torch.cat((embeddings, embedding.detach().clone()), dim=0)
        predictions = torch.cat((predictions, output.detach().clone()), dim=0)

        metric_logger.update(loss=loss.item())
        if args.downstream_task == 'classification':
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc = metric_acc(output.argmax(dim=-1), target)
            f1 = metric_f1(output.argmax(dim=-1), target)[args.pos_label]
            auroc = metric_auroc(torch.nn.functional.softmax(output, dim=-1)[:, args.pos_label], target)
            auprc = sklearn.metrics.average_precision_score(y_true=target.detach().cpu(), y_score=torch.nn.functional.softmax(output.detach().cpu().type(torch.float32), dim=-1)[:, args.pos_label], average='micro', pos_label=args.pos_label)
            # store the results of each step in a list to calculate the auc globally of the entire epoch
            [preds.append(elem.item()) for elem in torch.nn.functional.softmax(output, dim=-1)[:, args.pos_label]]
            [trgts.append(elem.item()) for elem in target]

            # my_target = target
            # my_predic = torch.argmax(output, dim=1)

            # # if pos_label=1
            # tp += (my_target * my_predic).sum()
            # fp += ((1-my_target) * my_predic).sum()
            # tn += ((1-my_target) * (1-my_predic)).sum()
            # fn += (my_target * (1-my_predic)).sum()

            # # print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")

            batch_size = images.shape[0]
            # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['acc'].update(100*acc.item(), n=batch_size)
            metric_logger.meters['f1'].update(100*f1.item(), n=batch_size)
            metric_logger.meters['auroc'].update(100*auroc.item(), n=batch_size)
            metric_logger.meters['auprc'].update(100*auprc.item(), n=batch_size)
        if args.downstream_task == 'regression':
            rmse = metric_rmse(output, target)
            pcc = metric_pcc(output, target)

            batch_size = images.shape[0]
            metric_logger.meters['rmse'].update(torch.tensor(rmse).mean().item(), n=batch_size)
            metric_logger.meters['pcc'].update(torch.tensor(pcc).mean().item(), n=batch_size)

    if args.wandb and args.plot_attention_map:
        idx = 1 if args.batch_size > 1 else 0
        plot.plot_attention(images, attention_map, idx)

    if args.predictions_dir:
        predictions_path = os.path.join(args.predictions_dir, "predictions")
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        if args.eval:
            torch.save(predictions.detach().cpu(), os.path.join(predictions_path, f"predictions_test.pt"))
        else:
            torch.save(predictions.detach().cpu(), os.path.join(predictions_path, f"predictions_{epoch}.pt"))

    if args.embeddings_dir:
        embeddings_path = os.path.join(args.embeddings_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        if args.eval:
            torch.save(embeddings.detach().cpu(), os.path.join(embeddings_path, f"embeddings_test.pt"))
        else:
            torch.save(embeddings.detach().cpu(), os.path.join(embeddings_path, f"embeddings_{epoch}.pt"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.downstream_task == 'classification':
        acc = 100*metric_acc.compute()
        metric_acc.reset()
        test_stats["acc"] = acc.item()

        f1 = 100*metric_f1.compute()[args.pos_label] # returns the f1 score
        metric_f1.reset()
        test_stats["f1"] = f1.item()

        # reset the auc metric to calculate the auc globally of the entire epoch
        metric_auroc.reset()
        metric_auroc(torch.tensor(preds, dtype=torch.float), torch.tensor(trgts, dtype=torch.long))
        auroc = 100*metric_auroc.compute() # returns the auroc for both classes combined (see average="macro")
        metric_auroc.reset()
        test_stats["auroc"] = auroc.item()

        auprc = 100*sklearn.metrics.average_precision_score(y_true=trgts, y_score=preds, average='micro', pos_label=args.pos_label)
        test_stats["auprc"] = auprc.item()
    elif args.downstream_task == 'regression':
        rmse = metric_rmse.compute()
        metric_rmse.reset()
        test_stats["rmse"] = torch.tensor(rmse).mean().item()

        pcc = metric_pcc.compute()
        metric_pcc.reset()
        test_stats["pcc"] = torch.tensor(pcc).mean().item()

    if args.downstream_task == 'classification':
        print('* Acc@1 {top1.global_avg:.3f} F1 {f1:.3f} AUROC {auroc:.3f} AUPRC {auprc:.3f} loss {losses:.3f}'
            .format(top1=metric_logger.acc, f1=test_stats["f1"], auroc=test_stats["auroc"], auprc=test_stats["auprc"], losses=test_stats["loss"]))
    elif args.downstream_task == 'regression':
        print('* RMSE {rmse:.3f} PCC {pcc:.2f} loss {losses:.3f}'
            .format(rmse=test_stats["rmse"], pcc=test_stats["pcc"], losses=test_stats["loss"]))

    if log_writer is not None:
        if args.downstream_task == 'classification':
            # log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            #log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_acc', test_stats['acc'], epoch)
            log_writer.add_scalar('perf/test_f1', test_stats['f1'], epoch)
            log_writer.add_scalar('perf/test_auroc', test_stats['auroc'], epoch)
            log_writer.add_scalar('perf/test_auprc', test_stats['auprc'], epoch)
        elif args.downstream_task == 'regression':
            log_writer.add_scalar('perf/test_rmse', test_stats['rmse'], epoch)
            log_writer.add_scalar('perf/test_pcc', test_stats['pcc'], epoch)
        log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        if args.wandb == True:
            test_history = {'epoch' : epoch,
                                'test_loss' : test_stats['loss']}
            if args.downstream_task == 'classification':
                # test_history['test_acc1'] = test_stats['acc1']
                # test_history['test_acc5'] = test_stats['acc5']
                test_history['test_acc'] = test_stats['acc']
                test_history['test_f1'] = test_stats['f1']
                test_history['test_auroc'] = test_stats['auroc']
                test_history['test_auprc'] = test_stats['auprc']
            elif args.downstream_task == 'regression':
                test_history['test_rmse'] = test_stats['rmse']
                test_history['test_pcc'] = test_stats['pcc']

                for i in range(target.shape[-1]):
                    test_history[f'Test/RMSE/{i}'] = torch.tensor(rmse[i]).item()
                    test_history[f'Test/PCC/{i}'] = pcc[i].item()

            if args.plot_embeddings and epoch % 10 == 0:
                reducer = umap.UMAP(n_components=2, metric='euclidean')
                umap_proj = reducer.fit_transform(embeddings.cpu())
                
                cmap = matplotlib.cm.get_cmap('tab20') # for the colours

                # plt.figure()
                fig, ax = plt.subplots(figsize=(8, 8))

                for label in range(args.nb_classes):
                    indices = np.array(trgts)==label
                    ax.scatter(umap_proj[indices, 0], umap_proj[indices, 1], c=np.array(cmap(label*3)).reshape(1, 4), label=label, alpha=0.5)

                ax.legend(fontsize='large', markerscale=2)

                test_history["UMAP Embeddings"] = wandb.Image(fig)
                plt.close('all')

            # wandb.log(test_history)
    
    return test_stats, test_history