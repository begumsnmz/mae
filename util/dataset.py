import os
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from util.transforms import MinMaxScaling, Normalization, ArrayToTensor, ScalarToTensor, OneHotEncoding
from util.augmentations import Jitter, Masking, Rescaling, Permutation, Shift, TimeToFourier, CropResizing, Interpolation


class EEGDatasetFast(Dataset):
    """Fast EEGDataset (fetching prepared data and labels from files)"""
    def __init__(self, transform=False, augment=False, mode=None, args=None) -> None:
        """load data and labels from files"""
        self.transform = transform
        self.augment = augment
        self.mode = mode
        
        self.args = args

        self.data = torch.load(args.data_path, map_location=torch.device('cpu')) # load to ram
        self.labels = torch.load(args.labels_path, map_location=torch.device('cpu')) # load to ram

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        data, label = self.data[idx], self.labels[idx]
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=0)

        data = data[:, :self.args.input_electrodes]

        if self.transform == True:
            # transform = transforms.Compose([TimeToFourier(), 
            #                                 CropResizing(fixed_len=True)])
            # transform = Normalization()
            # transform = CropResizing(fixed_len=self.args.input_size[-1], start_idx=3000) 
            transform = CropResizing(fixed_len=self.args.input_size[-1], start_idx=0) # THIS IS ONLY FOR SEED
            data = transform(data)

        if self.augment == True:
            if self.mode is None:
                augment = transforms.Compose([Jitter(),
                                              Rescaling(),
                                              CropResizing(fixed_len=self.args.input_size[-1])])
            else:
                augment = transforms.Compose([Jitter(),
                                              Rescaling(),
                                              CropResizing(fixed_len=self.args.input_size[-1]),
                                              Permutation()])
            data = augment(data)

        return data, label.type(torch.LongTensor).argmax(dim=-1)