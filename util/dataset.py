import os
import sys
from typing import Any, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.transformations as transformations
import util.augmentations as augmentations


class EEGDatasetFast(Dataset):
    """Fast EEGDataset (fetching prepared data and labels from files)"""
    def __init__(self, transform=False, augment=False, transfer=False, args=None) -> None:
        """load data and labels from files"""
        self.transform = transform
        self.augment = augment
        
        self.args = args

        if transfer == False:
            self.data = torch.load(args.data_path, map_location=torch.device('cpu')) # load to ram
            self.labels = torch.load(args.labels_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.data = torch.load(args.transfer_data_path, map_location=torch.device('cpu')) # load to ram
            self.labels = torch.load(args.transfer_labels_path, map_location=torch.device('cpu')) # load to ram

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        data, label = self.data[idx], self.labels[idx]
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=0)

        data = data[:, :self.args.input_electrodes, :]
        
        if self.transform == True:
            transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0)
            data = transform(data)

        if self.augment == True:
            lower_bnd = self.args.crop_lbd * self.args.input_size[-1] / data.shape[-1]
            upper_bnd = 1.00 * self.args.input_size[-1] / data.shape[-1]
            augment = transforms.Compose([augmentations.Jitter(sigma=self.args.jitter_sigma),
                                          augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                                          augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                                          augmentations.CropResizing(lower_bnd=lower_bnd, upper_bnd=upper_bnd, resize=True, fixed_resize_len=self.args.input_size[-1]),
                                          #augmentations.TimeFlip(prob=0.33),
                                          #augmentations.SignFlip(prob=0.33)
                                          ])
            data = augment(data)

        # label = self.labels[idx]
        # data = torch.sin(math.pi*torch.linspace(0, 4, self.args.input_size[-1])).unsqueeze(dim=0).repeat(65, 1).unsqueeze(dim=0) \
        #         + torch.sin(math.pi*torch.linspace(0, 8, self.args.input_size[-1])).unsqueeze(dim=0).repeat(65, 1).unsqueeze(dim=0) \
        #         + torch.sin(math.pi*torch.linspace(0, 16, self.args.input_size[-1])).unsqueeze(dim=0).repeat(65, 1).unsqueeze(dim=0)

        return data, label.type(torch.LongTensor).argmax(dim=-1)