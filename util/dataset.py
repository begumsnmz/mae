import os
import sys
from typing import Any, Tuple

import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.transformations as transformations
import util.augmentations as augmentations


class SignalDataset(Dataset):
    """
    Unimodal dataset that generates views of signals.
    """
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task=None, 
                 train=False, args=None) -> None:
        """load data and labels from files"""
        self.downstream_task = downstream_task
        
        self.train = train 
        
        self.args = args

        #data_paths = glob.glob(os.path.join(data_path, '*.pt'))
        #data = torch.stack([torch.tensor(torch.load(path, map_location=torch.device('cpu')), dtype=torch.float32) for path in data_paths])

        #Single Tensor
        data = torch.load(data_path, map_location=torch.device('cpu')) # load to ram
        data = data.to(torch.float32)
    
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=1)

        #Index first args.input_electrodes 
        self.data = data[..., :self.args.input_electrodes, :]
            
        #Randperm electrodes
        #shuffle_idx = torch.randperm(self.args.input_electrodes)
        #self.data = data[..., shuffle_idx, :]
            
        #Select common electrodes (LEMON and TUAB)
        #common_ch = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 28, 30, 40, 43]
        #self.data = data[..., common_ch, :]
        #print(self.data.size())


        #Overfit Case
        if args.overfit == True:
            if self.train == True: #Training samples
                rand_idx =  torch.randperm(self.data.shape[0])
                self.data = self.data[rand_idx[:args.overfit_sample_size], :, :, :]
                #print(self.data.size())
            else: #Validation samples
                self.data = self.data[:10, :, :, :]
        else:
            self.data = self.data
    

        if labels_path:
            if args.overfit == True:
                if self.train == True:
                    labels_unsq = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
                    self.labels = labels_unsq[rand_idx[:args.overfit_sample_size]].unsqueeze(dim=1)
                    print(self.labels)
                else:
                    labels_unsq = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
                    self.labels = labels_unsq[:10].unsqueeze(dim=1)
                    print(self.labels)
            else:
                labels_unsq = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
                self.labels = labels_unsq.unsqueeze(dim=1)
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)


    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]

        #NORMALIZE AND CLAMP DATA
        data = (data - data.mean(-1, keepdims=True)) / (data.std(-1, keepdims=True) + 1e-8)
        data = torch.clamp(data, min=-20, max=20)

        if self.train == False:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
        else:
            if self.args.overfit == True: #No RANDOM cropping for Overfitting
                transform = transforms.Compose([
                    augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                ])
            else:
                transform = transforms.Compose([
                    augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                ])


        data = transform(data)
        
        if self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        return data, label, label_mask
    