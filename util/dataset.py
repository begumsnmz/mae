import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import util.augmentations as augmentations
import numpy as np


class SignalDataset(Dataset):
    """
    Unimodal dataset that generates views of signals, supporting both single tensor and individual file modes.
    """
    def __init__(self, data_path, labels_path=None, label_map_path=None, labels_mask_path=None,
                 downstream_task=None, train=False, indices=None, args=None):
        """Load data and labels from files"""
        self.downstream_task = downstream_task
        self.train = train
        self.args = args

        self.label_map = {}


        # Determine if data_path points to a single .pt file or a directory of .pt files
        if os.path.isfile(data_path) and data_path.endswith('.pt'):
            # Case 1: Single tensor file
            self.data = torch.load(data_path, map_location=torch.device('cpu')).to(torch.float32)
            if self.args.input_size[0] == 1:
                self.data = self.data.unsqueeze(dim=1)
            if self.args.electrode_idx is not None:
                # Slice using the provided indices
                self.data = self.data[..., self.args.electrode_idx, :]
            else:
                # Slice using another method if no indices are provided
                self.data = self.data[..., :self.args.input_electrodes, :]
            self.mode = 'tensor'
            #Overfit Case
            if args.overfit == True:
                if self.train == True: #Training samples
                    rand_idx =  torch.randperm(self.data.shape[0])
                    self.data = self.data[rand_idx[:args.overfit_sample_size], ...]
                else: #Validation samples
                    self.data = self.data[:10, ...]
            else:
                self.data = self.data
        else:
            # Case 2: Directory of individual .pt files
            self.file_paths = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')])
            self.mode = 'files'
            # Apply percentage sampling
            if args.scale_percentage != 100:
                total_samples = len(self.file_paths)
                sample_count = int(total_samples * (args.scale_percentage / 100.0))
                np.random.seed(42)
                self.file_paths = np.random.choice(self.file_paths, sample_count, replace=False)


        # Load labels or label map
        if label_map_path:
            self.label_map = torch.load(label_map_path, map_location=torch.device('cpu'))
            self.label_map = {key: torch.tensor(value) if not isinstance(value, torch.Tensor) else value for key, value in self.label_map.items()}
            self.labels = torch.stack([self.label_map[os.path.basename(fp)] for fp in self.file_paths])
        elif labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))
        else:
            if self.file_paths:
                self.labels = torch.zeros(size=(len(self.file_paths), ))
            else:
                self.labels = torch.zeros(size=(len(self.data), ))


        # Overfit Case
        if args.overfit:
            if self.train:
                self.labels = self.labels[rand_idx[:args.overfit_sample_size]]
            else:
                self.labels = self.labels[:10]

        self.labels = self.labels.unsqueeze(dim=1) if self.labels.dim() == 1 else self.labels

        # Load labels mask if provided
        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu'))
        else:
            self.labels_mask = torch.ones_like(self.labels) if self.labels is not None else None


        # Subset the data if indices are provided
        if indices is not None:
            if self.mode == 'tensor':
                self.data = self.data[indices]
            else:
                self.file_paths = [self.file_paths[i] for i in indices]

            self.labels = self.labels[indices]
            self.labels_mask = self.labels_mask[indices]

    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.mode == 'tensor':
            return self.data.size(0)
        else:
            return len(self.file_paths)

    def __getitem__(self, idx):
        """Return a sample from the dataset at index idx"""
        if self.mode == 'tensor':
            data = self.data[idx]
            label = self.labels[idx]
            label_mask = self.labels_mask[idx]
        else:
            # Load data from file path
            data_dict = torch.load(self.file_paths[idx], map_location=torch.device('cpu'))
            data = data_dict["data"].to(torch.float32)
            if self.args.input_size[0] == 1:
                data = data.unsqueeze(dim=0)

            #Electrode indexing if necessary
            if self.args.electrode_idx is not None:
                # Slice using the provided indices
                data = data[..., self.args.electrode_idx, :]
            else:
                data = data[..., :self.args.input_electrodes, :]

            #filename = os.path.basename(self.file_paths[idx])
            #label = torch.tensor(self.label_map.get(filename, torch.tensor(0)))
            #label = label.unsqueeze(dim=1)
            label = self.labels[idx]
            label_mask = self.labels_mask[idx] if self.labels_mask is not None else torch.tensor(1)

        # Handle downstream tasks
        if self.downstream_task == 'regression':
            label = label[..., self.args.lower_bnd:self.args.upper_bnd] if isinstance(label, torch.Tensor) else label
            label_mask = label_mask[..., self.args.lower_bnd:self.args.upper_bnd] if isinstance(label_mask, torch.Tensor) else label_mask
        elif self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        # Normalize and clamp data
        data = (data - data.mean(-1, keepdims=True)) / (data.std(-1, keepdims=True) + 1e-8)
        data = torch.clamp(data, min=-20, max=20)

        if self.train == False:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
            ])
            #print("I am here at validation")
        else:
            if self.args.overfit == True: #No RANDOM cropping for Overfitting
                transform = transforms.Compose([
                    augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                ])
            else:
                transform = transforms.Compose([
                    augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                ])
                #print("I am here at training")

        data = transform(data)

        return data, label, label_mask


