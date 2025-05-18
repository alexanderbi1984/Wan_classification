import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Optional
import yaml
from collections import Counter
from sklearn.model_selection import StratifiedKFold

class SyracuseMultimodalDataset(Dataset):
    """
    Multimodal dataset for Syracuse: loads VAE and xDiT features from separate folders using the same filename.
    Returns (vae_x, xdit_x, pain_label, -1)
    Supports temporal_pooling and flatten as in the single-modality version.
    """
    def __init__(self, vae_feature_dir, xdit_feature_dir, meta_path, task="classification", thresholds=None, transform=None, set_name=None, temporal_pooling='mean', flatten=True, source_filter=None, video_ids=None):
        self.vae_feature_dir = vae_feature_dir
        self.xdit_feature_dir = xdit_feature_dir
        self.meta_path = meta_path
        self.transform = transform
        self.set_name = set_name
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.task = task
        self.thresholds = thresholds
        self.source_filter = source_filter if source_filter is not None else ['syracuse_original', 'syracuse_aug']
        self.video_ids = set(video_ids) if video_ids is not None else None
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.samples = []
        for fname, entry in meta.items():
            src = entry.get('source')
            p = entry.get('pain_level')
            vid = entry.get('video_id')
            if src not in self.source_filter or p is None:
                continue
            if self.video_ids is not None and vid not in self.video_ids:
                continue
            label = self.get_class_label(p)
            self.samples.append({
                'fname': fname,
                'pain_level': float(p),
                'class_label': label,
                'video_id': vid,
                'clip_id': entry.get('clip_id'),
                'source': src
            })
        self.samples.sort(key=lambda x: x['fname'])
        self._example_shape = None
        if self.samples:
            arr_vae = torch.from_numpy(np.load(os.path.join(self.vae_feature_dir, self.samples[0]['fname']))).float()
            arr_xdit = torch.from_numpy(np.load(os.path.join(self.xdit_feature_dir, self.samples[0]['fname']))).float()
            arr_vae = self._apply_pooling(arr_vae)
            arr_xdit = self._apply_pooling(arr_xdit)
            if self.flatten:
                arr_vae = arr_vae.view(-1)
                arr_xdit = arr_xdit.view(-1)
            self._example_shape = (tuple(arr_vae.shape), tuple(arr_xdit.shape))

    def get_class_label(self, pain):
        if self.task != 'classification' or self.thresholds is None:
            return None
        t = self.thresholds
        if len(t) == 0:
            return int(pain > 0)
        for i, th in enumerate(t):
            if pain <= th:
                return i
        return len(t)

    def _apply_pooling(self, arr):
        if self.temporal_pooling == 'mean':
            arr = arr.mean(dim=1)
        elif self.temporal_pooling == 'max':
            arr = arr.max(dim=1).values
        elif self.temporal_pooling == 'sample':
            T_target = 4
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
        elif self.temporal_pooling == 'none':
            T_target = 4
            T_actual = arr.shape[1]
            if T_actual == T_target:
                arr = arr
            elif T_actual > T_target:
                arr = arr[:, :T_target, ...]
            else:
                pad_shape = list(arr.shape)
                pad_shape[1] = T_target - T_actual
                pad = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
                arr = torch.cat([arr, pad], dim=1)
        else:
            raise ValueError(f"Invalid temporal_pooling: {self.temporal_pooling}")
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        arr_vae = torch.from_numpy(np.load(os.path.join(self.vae_feature_dir, x['fname']))).float()
        arr_xdit = torch.from_numpy(np.load(os.path.join(self.xdit_feature_dir, x['fname']))).float()
        arr_vae = self._apply_pooling(arr_vae)
        arr_xdit = self._apply_pooling(arr_xdit)
        if self.flatten:
            arr_vae = arr_vae.view(-1)
            arr_xdit = arr_xdit.view(-1)
        if self.transform:
            arr_vae = self.transform(arr_vae)
            arr_xdit = self.transform(arr_xdit)
        assert arr_vae.ndim == 4, f"VAE feature shape is {arr_vae.shape}, expected 4D (C, T, H, W)"
        # print(f"idx={idx}, vae_x shape={arr_vae.shape}, xdit_x shape={arr_xdit.shape}")
        pain_label = x['class_label'] if self.task == 'classification' else x['pain_level']
        return arr_vae, arr_xdit, torch.tensor(pain_label, dtype=torch.long), torch.tensor(-1, dtype=torch.long)

    @property
    def example_shape(self):
        return self._example_shape

class BioVidMultimodalDataset(Dataset):
    """
    Multimodal dataset for BioVid: loads VAE and xDiT features from separate folders using the same filename.
    Returns (vae_x, xdit_x, -1, stim_label)
    Supports temporal_pooling and flatten as in the single-modality version.
    """
    def __init__(self, vae_feature_dir, xdit_feature_dir, meta_path, transform=None, subject_ids=None, temporal_pooling='mean', flatten=True):
        self.vae_feature_dir = vae_feature_dir
        self.xdit_feature_dir = xdit_feature_dir
        self.meta_path = meta_path
        self.transform = transform
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.samples = []
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
        self.samples.sort(key=lambda x: x['fname'])
        self._example_shape = None
        if self.samples:
            arr_vae = torch.from_numpy(np.load(os.path.join(self.vae_feature_dir, self.samples[0]['fname']))).float()
            arr_xdit = torch.from_numpy(np.load(os.path.join(self.xdit_feature_dir, self.samples[0]['fname']))).float()
            arr_vae = self._apply_pooling(arr_vae)
            arr_xdit = self._apply_pooling(arr_xdit)
            if self.flatten:
                arr_vae = arr_vae.view(-1)
                arr_xdit = arr_xdit.view(-1)
            self._example_shape = (tuple(arr_vae.shape), tuple(arr_xdit.shape))

    def _apply_pooling(self, arr):
        if self.temporal_pooling == 'mean':
            arr = arr.mean(dim=1)
        elif self.temporal_pooling == 'max':
            arr = arr.max(dim=1).values
        elif self.temporal_pooling == 'sample':
            T_target = 4
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
        elif self.temporal_pooling == 'none':
            T_target = 4
            T_actual = arr.shape[1]
            if T_actual == T_target:
                arr = arr
            elif T_actual > T_target:
                arr = arr[:, :T_target, ...]
            else:
                pad_shape = list(arr.shape)
                pad_shape[1] = T_target - T_actual
                pad = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
                arr = torch.cat([arr, pad], dim=1)
        else:
            raise ValueError(f"Invalid temporal_pooling: {self.temporal_pooling}")
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        arr_vae = torch.from_numpy(np.load(os.path.join(self.vae_feature_dir, x['fname']))).float()
        arr_xdit = torch.from_numpy(np.load(os.path.join(self.xdit_feature_dir, x['fname']))).float()
        arr_vae = self._apply_pooling(arr_vae)
        arr_xdit = self._apply_pooling(arr_xdit)
        if self.flatten:
            arr_vae = arr_vae.view(-1)
            arr_xdit = arr_xdit.view(-1)
        if self.transform:
            arr_vae = self.transform(arr_vae)
            arr_xdit = self.transform(arr_xdit)
        assert arr_vae.ndim == 4, f"VAE feature shape is {arr_vae.shape}, expected 4D (C, T, H, W)"
        # print(f"idx={idx}, vae_x shape={arr_vae.shape}, xdit_x shape={arr_xdit.shape}")
        return arr_vae, arr_xdit, torch.tensor(-1, dtype=torch.long), torch.tensor(x['label'], dtype=torch.long)

    @property
    def example_shape(self):
        return self._example_shape 

class SyracuseMultimodalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Syracuse multimodal dataset with pain level as ground truth and YAML-configurable classification.
    Provides cross-validation splits and optional balanced sampling for classification.
    Mirrors the anti-leakage, video-aware, stratified split logic of SyracuseDataModule in syracuse.py.
    """
    def __init__(self, vae_feature_dir, xdit_feature_dir, meta_path, task: str = "classification", thresholds: list = None, batch_size=32, num_workers=4, cv_fold=3, seed=42,
                 balanced_sampling=False, transform=None, temporal_pooling='mean', flatten=True):
        super().__init__()
        self.vae_feature_dir = vae_feature_dir
        self.xdit_feature_dir = xdit_feature_dir
        self.meta_path = meta_path
        self.task = task
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cv_fold = cv_fold
        self.seed = seed
        self.balanced_sampling = balanced_sampling
        self.transform = transform
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        meta_dir = os.path.dirname(self.meta_path)
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        originals = [e for e in meta.values() if e.get('source') == 'syracuse_original' and e.get('pain_level') is not None]
        orig_video_ids = sorted({e['video_id'] for e in originals})
        pain_per_vid = {e['video_id']: e['pain_level'] for e in originals}

        if self.task == 'classification':
            if self.thresholds is None:
                raise ValueError("Thresholds must be provided for classification task in SyracuseMultimodalDataModule.")
            current_thresholds = self.thresholds
            def label_fn(pain):
                for i, th in enumerate(current_thresholds):
                    if pain <= th:
                        return i
                return len(current_thresholds)
            split_vec = [label_fn(pain_per_vid[vid]) for vid in orig_video_ids]
            skf = StratifiedKFold(n_splits=3, random_state=self.seed, shuffle=True)
        elif self.task == 'regression':
            n_bins = 5
            bin_edges = np.linspace(0, 10, n_bins+1)[1:]
            def pseudo_label(pain):
                for i, th in enumerate(bin_edges):
                    if pain <= th:
                        return i
                return n_bins - 1
            split_vec = [pseudo_label(pain_per_vid[vid]) for vid in orig_video_ids]
            skf = StratifiedKFold(n_splits=3, random_state=self.seed, shuffle=True)
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

        train_vids, val_vids = None, None
        if self.cv_fold in [0, 1, 2]:
            for fold_idx, (train_index, val_index) in enumerate(skf.split(orig_video_ids, split_vec)):
                if fold_idx == self.cv_fold:
                    train_vids = set([orig_video_ids[i] for i in train_index])
                    val_vids   = set([orig_video_ids[i] for i in val_index])
                    break
        else:
            train_vids = set(orig_video_ids)
            val_vids = set()
        print(f"[DEBUG] train_vids: {len(train_vids)}; sample: {list(train_vids)[:5]}")
        print(f"[DEBUG] val_vids: {len(val_vids)}; sample: {list(val_vids)[:5]}")
        self.train_dataset = SyracuseMultimodalDataset(
            self.vae_feature_dir,
            self.xdit_feature_dir,
            self.meta_path,
            task=self.task,
            thresholds=self.thresholds,
            transform=self.transform,
            set_name="TRAIN",
            temporal_pooling=self.temporal_pooling,
            flatten=self.flatten,
            source_filter=["syracuse_original", "syracuse_aug"],
            video_ids=train_vids
        )
        if self.cv_fold < 3:
            self.val_dataset = SyracuseMultimodalDataset(
                self.vae_feature_dir,
                self.xdit_feature_dir,
                self.meta_path,
                task=self.task,
                thresholds=self.thresholds,
                transform=self.transform,
                set_name="VAL",
                temporal_pooling=self.temporal_pooling,
                flatten=self.flatten,
                source_filter=["syracuse_original"],
                video_ids=val_vids
            )
        else:
            self.val_dataset = None
        # Optionally print class distributions
        train_labels = [s['class_label'] for s in self.train_dataset.samples if 'class_label' in s]
        val_labels = [s['class_label'] for s in self.val_dataset.samples if 'class_label' in s] if self.val_dataset else []
        print(f"[INFO] Train class dist: {Counter(train_labels)}")
        print(f"[INFO] Val class dist: {Counter(val_labels)}")

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        is_classification = (self.train_dataset.task == 'classification')
        if self.balanced_sampling:
            if not is_classification:
                raise RuntimeError("Balanced sampling is only applicable to classification tasks.")
            labels = [s['class_label'] for s in self.train_dataset.samples]
            if not all(l is not None for l in labels):
                raise ValueError("Can't use balanced sampling: found None in class labels.")
            label_counts = Counter(labels)
            class_weights = {cls: 1.0 / cnt for cls, cnt in label_counts.items()}
            sample_weights = [class_weights[s['class_label']] for s in self.train_dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                              sampler=sampler)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, generator=generator)

    def val_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, generator=generator)

    def __repr__(self):
        s = f"<SyracuseMultimodalDataModule: batch_size={self.batch_size}, num_workers={self.num_workers}, cv_fold={self.cv_fold}, config={os.path.basename(self.meta_path)}>\n"
        if hasattr(self, 'train_dataset'):
            s += f"Train: {self.train_dataset}\n"
        if hasattr(self, 'val_dataset'):
            s += f"Val: {self.val_dataset}"
        return s

    @property
    def example_shape(self):
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return self.train_dataset.example_shape
        else:
            return None 