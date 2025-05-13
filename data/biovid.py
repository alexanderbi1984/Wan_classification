import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
import random
from collections import Counter

class BioVidDataset(Dataset):
    def __init__(self, features_path, meta_path, subject_ids=None, transform=None, use_video_fallback=False, temporal_pooling='mean', flatten=True, input_format='CTHW'):
        """
        Args:
            features_path: Path to npy feature folder
            meta_path: Path to meta.json
            subject_ids: list/set of subject ids to include
            transform: Optional transforms
            use_video_fallback: (unused)
            temporal_pooling: 'mean', 'max', or 'sample'
                'sample' will randomly select 5 frames (pad with zeros if not enough)
            flatten: If True, return features as 1D. If False, return spatial/temporal shape as processed.
            input_format: Format of input features, either 'CTHW' for [C,T,H,W] or 'TCHW' for [T,C*H*W]
        """
        assert input_format in ['CTHW', 'TCHW'], f"input_format must be 'CTHW' or 'TCHW', got {input_format}"
        self.features_path = features_path
        self.transform = transform
        self.use_video_fallback = use_video_fallback
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.input_format = input_format
        self.samples = []
        # Read meta.json and filter by biovid + subject_ids
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        for fname, entry in meta.items():
            if entry.get('source') == 'biovid':
                subject_id = entry['subject_id']
                label = entry['5_class_label']
                if (subject_ids is None) or (subject_id in subject_ids):
                    self.samples.append({
                        'fname': fname,
                        'subject_id': subject_id,
                        'label': label,
                    })
        self.samples.sort(key=lambda x: x['fname'])  # Deterministic order
        self._example_shape = None
        if self.samples:
            arr = np.load(os.path.join(self.features_path, self.samples[0]['fname']))
            arr = torch.from_numpy(arr).float()
            
            # Handle different input formats
            if self.input_format == 'TCHW':
                if arr.ndim != 2:
                    raise ValueError(f"TCHW format expects 2D tensor [T,D], got shape {arr.shape}")
                T, D = arr.shape
                # Fixed C=3, calculate H=W
                C = 3
                remaining = D // C
                H = int(np.sqrt(remaining))
                if H * H != remaining:
                    raise ValueError(f"Cannot reshape tensor with D={D} into square spatial dimensions with C={C}")
                arr = arr.view(T, C, H, H)
                arr = arr.permute(1, 0, 2, 3)  # [C,T,H,W]
            elif self.input_format == 'CTHW':
                if arr.ndim != 4:
                    raise ValueError(f"CTHW format expects 4D tensor [C,T,H,W], got shape {arr.shape}")
            
            # Apply pooling logic for true shape after processing
            if self.temporal_pooling == 'mean':
                arr = arr.mean(dim=1)
            elif self.temporal_pooling == 'max':
                arr = arr.max(dim=1).values
            elif self.temporal_pooling == 'sample':
                T_target = 5
                T_actual = arr.shape[1]
                if T_actual == T_target:
                    arr = arr
                elif T_actual > T_target:
                    start = torch.randint(0, T_actual - T_target + 1, (1,)).item()
                    arr = arr[:, start:start+T_target, ...]
                else:
                    pad_shape = list(arr.shape)
                    pad_shape[1] = T_target - T_actual
                    pad = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
                    arr = torch.cat([arr, pad], dim=1)
            if self.flatten:
                arr = arr.view(-1)
            self._example_shape = tuple(arr.shape)

    @property
    def example_shape(self):
        """
        Shape of each processed feature sample (after temporal pooling and flattening if enabled), e.g. (32768,) or (16, 5, 128, 128)
        """
        return self._example_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npy_path = os.path.join(self.features_path, sample['fname'])
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            arr = torch.from_numpy(arr).float()
            
            # Handle different input formats
            if self.input_format == 'TCHW':
                if arr.ndim != 2:
                    raise ValueError(f"TCHW format expects 2D tensor [T,D], got shape {arr.shape}")
                T, D = arr.shape
                # Fixed C=3, calculate H=W
                C = 3
                remaining = D // C
                H = int(np.sqrt(remaining))
                if H * H != remaining:
                    raise ValueError(f"Cannot reshape tensor with D={D} into square spatial dimensions with C={C}")
                arr = arr.view(T, C, H, H)
                arr = arr.permute(1, 0, 2, 3)  # [C,T,H,W]
            elif self.input_format == 'CTHW':
                if arr.ndim != 4:
                    raise ValueError(f"CTHW format expects 4D tensor [C,T,H,W], got shape {arr.shape}")
            
            # --- Temporal Pooling ---
            if self.temporal_pooling == 'mean':
                arr = arr.mean(dim=1)  # [C,H,W]
            elif self.temporal_pooling == 'max':
                arr = arr.max(dim=1).values # [C,H,W]
            elif self.temporal_pooling == 'sample':
                T_target = 5
                T_actual = arr.shape[1]
                if T_actual == T_target:
                    arr = arr
                elif T_actual > T_target:
                    start = torch.randint(0, T_actual - T_target + 1, (1,)).item()
                    arr = arr[:, start:start+T_target, ...]
                else:
                    # Pad with zeros at the end
                    pad_shape = list(arr.shape)
                    pad_shape[1] = T_target - T_actual
                    pad = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
                    arr = torch.cat([arr, pad], dim=1)
            else:
                raise ValueError(f"Invalid temporal_pooling: {self.temporal_pooling}")
            if self.flatten:
                arr = arr.view(-1)
        else:
            if self.use_video_fallback:
                basename = os.path.splitext(sample['fname'])[0]
                for ext in ('mp4', 'avi'):
                    vid_path = os.path.join(self.features_path, f"{basename}.{ext}")
                    if os.path.exists(vid_path):
                        raise NotImplementedError('Video loading not implemented yet.')
                raise FileNotFoundError(f"No npy or video file for {sample['fname']}")
            else:
                raise FileNotFoundError(f"Feature file not found: {npy_path}")
        if self.transform:
            arr = self.transform(arr)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return arr, sample['subject_id'], label

    def label_distribution(self):
        return Counter([s['label'] for s in self.samples])

    def __repr__(self):
        s = f"<BioVidDataset: {len(self)} samples"
        if self.example_shape:
            s += f", sample shape={self.example_shape}"
        s += ">"
        return s

class BioVidDataModule(pl.LightningDataModule):
    def __init__(self, features_path, meta_path, batch_size=32, num_workers=4, split_ratio=0.8, seed=42, use_video_fallback=False, temporal_pooling='mean', flatten=True, input_format='CTHW'):
        super().__init__()
        self.features_path = features_path
        self.meta_path = meta_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.seed = seed
        self.use_video_fallback = use_video_fallback
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.input_format = input_format

    def setup(self, stage: Optional[str] = None):
        # Gather all biovid subject_ids
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        subject_ids = sorted({entry['subject_id'] for entry in meta.values() if entry.get('source') == 'biovid'})
        # Deterministic shuffle by seed
        rng = random.Random(self.seed)
        subject_ids_shuffled = subject_ids[:]
        rng.shuffle(subject_ids_shuffled)
        n_train = int(len(subject_ids_shuffled) * self.split_ratio)
        train_subjects = set(subject_ids_shuffled[:n_train])
        val_subjects = set(subject_ids_shuffled[n_train:])
        # Build datasets
        self.train_dataset = BioVidDataset(
            self.features_path, self.meta_path, subject_ids=train_subjects,
            use_video_fallback=self.use_video_fallback, temporal_pooling=self.temporal_pooling,
            flatten=self.flatten, input_format=self.input_format)
        self.val_dataset = BioVidDataset(
            self.features_path, self.meta_path, subject_ids=val_subjects,
            use_video_fallback=self.use_video_fallback, temporal_pooling=self.temporal_pooling,
            flatten=self.flatten, input_format=self.input_format)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    @property
    def example_shape(self):
        """
        Shape of feature for model input, provided by train_dataset.
        """
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return self.train_dataset.example_shape
        else:
            return None

    def __repr__(self):
        s = f"<BioVidDataModule: batch_size={self.batch_size}, num_workers={self.num_workers}, split_ratio={self.split_ratio}, seed={self.seed}>\n"
        if hasattr(self, 'train_dataset'):
            s += f"Train: {self.train_dataset}\n"
            dist = self.train_dataset.label_distribution()
            s += f"  Train label dist: {dict(dist)}\n"
        if hasattr(self, 'val_dataset'):
            s += f"Val: {self.val_dataset}\n"
            dist = self.val_dataset.label_distribution()
            s += f"  Val label dist: {dict(dist)}"
        return s

# Example usage:
# dm = BioVidDataModule(features_path="/home/nbi/marlin/wan_features/", meta_path="/home/nbi/marlin/wan_features/meta.json")
# dm.setup()
# print(dm) 