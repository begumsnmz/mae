import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import util.augmentations as augmentations
import numpy as np


class SignalDataset(Dataset):
    """
    Unimodal dataset that generates views of signals, supporting both single tensor, individual file modes
    and memory mapped arrays.
    """

    def __init__(self, data_path, labels_path=None, label_map_path=None, labels_mask_path=None, subject_file_path=None,
                 downstream_task=None, train=False, indices=None, shape=None, num_chunks=None, args=None):
        """Load data and labels from files"""
        self.downstream_task = downstream_task
        self.train = train
        self.args = args
        self.label_map = {}
        self.mode = None
        self.num_chunks = num_chunks

        # Handle different data sources
        if os.path.isfile(data_path) and data_path.endswith('.pt'):
            # Case 1: Single tensor file
            # Use torch.load with mmap_mode
            self.mode = 'tensor'
            self.data = torch.load(data_path, map_location='cpu').to(torch.float32)
        elif os.path.isfile(data_path) and data_path.endswith('.dat'):
            # Case 2: Memory-mapped numpy file
            self.mode = 'npy'
            self.data = np.memmap(data_path, dtype='float32', mode='r', shape=shape)
            if subject_file_path:
                self.subjects = np.load(subject_file_path)
        else:
            # Case 3: Directory of individual .pt files
            self.file_paths = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')])
            self.mode = 'files'

        # Apply percentage sampling
        if args.scale_percentage != 100:
                total_samples = self.data.size(0) if self.mode == 'tensor' else self.data.shape[0] if self.mode == 'npy' else len(self.file_paths)
                sample_count = int(total_samples * (args.scale_percentage / 100.0))
                np.random.seed(13)
                idxs = np.random.choice(np.arange(total_samples), size=sample_count, replace=False)
                if self.mode == 'tensor':
                    self.data = self.data[idxs,:,:]
                elif self.mode == 'npy':
                    self.data = self.data[idxs, :, :]
                    if subject_file_path:
                        self.subjects = self.subjects[indices]
                else:
                    self.file_paths = self.file_paths[idxs]

        # LABELS
        # Load labels or label map
        if label_map_path:
            self.label_map = torch.load(label_map_path, map_location=torch.device('cpu'))
            self.label_map = {key: torch.tensor(value) if not isinstance(value, torch.Tensor) else value for key, value in self.label_map.items()}
            self.labels = torch.stack([self.label_map[os.path.basename(fp)] for fp in self.file_paths])
        elif labels_path:
            if labels_path.endswith('.pt'):
                self.labels = torch.load(labels_path, map_location=torch.device('cpu'))
            else:
                self.labels = torch.tensor(np.load(labels_path), dtype=torch.float32, device=torch.device('cpu'))
        else:
            if self.mode == 'files':
                self.labels = torch.zeros(size=(len(self.file_paths), ))
            else:
                self.labels = torch.zeros(size=(self.data.shape[0], ))

        self.labels = self.labels.unsqueeze(dim=1) if self.labels.dim() == 1 else self.labels

        # LABELS MASK
        # Load labels mask if provided
        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu'))
        else:
            self.labels_mask = torch.ones_like(self.labels) if self.labels is not None else None


        # Subset the data if indices are provided
        if indices is not None:
            if self.mode == 'tensor':
                self.data = self.data[indices]
            elif self.mode == 'npy':
                self.data = self.data[indices]
            else:
                self.file_paths = [self.file_paths[i] for i in indices]

            self.labels = self.labels[indices]
            self.labels_mask = self.labels_mask[indices]


    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.data.shape[0] if self.mode in ['tensor', 'npy'] else len(self.file_paths)


    def __getitem__(self, idx):
        """Return a sample from the dataset at index idx"""
        if self.mode == 'tensor':
            data = self.data[idx]
        elif self.mode == 'npy':
            data = torch.tensor(self.data[idx], dtype=torch.float32)
        else:
            data_dict = torch.load(self.file_paths[idx], map_location=torch.device('cpu'))
            data = data_dict["data"].to(torch.float32)

        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=0)
        if self.args.electrode_idx is not None:
            data = data[..., self.args.electrode_idx, :]
        else:
            data = data[..., :self.args.input_electrodes, :]

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

        if self.train == False: #VALIDATION
            #transform = transforms.Compose([augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False, sequential=True, args=self.args),])
            #Chunk the data
            chunks = torch.split(data, self.args.input_size[-1], dim=-1)
            if chunks[-1].size(-1) != self.args.input_size[-1]:
                chunks = chunks[:-1] #Drop the last chunk if its smaller than time_steps
            if self.num_chunks: #If the argument is provided, take only the first n chunks
                chunks = chunks[:self.num_chunks]
            return [(chunk, label, label_mask) for chunk in chunks]

        else:  # TRAINING
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
            ])
            data = transform(data)
            return data, label, label_mask

    def get_subjects(self):
        return self.subjects

