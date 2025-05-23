"""
Syracuse dataset and PyTorch Lightning DataModule for pain-level classification/regression.
Supports YAML-configurable thresholds, stratified video-aware splits, and augmentation handling for research pipelines.
"""
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

class SyracuseDataset(Dataset):
    """
    PyTorch dataset for Syracuse video features with pain levels and optional class label thresholds.
    Args:
        temporal_pooling: 'mean', 'max', 'sample', or 'none'.
            'sample' will randomly select 4 frames (pad with zeros if not enough).
            'none' will keep the temporal dimension (no pooling).
    """
    def __init__(self, meta_path, source_filter, video_ids=None, task="classification", thresholds=None, transform=None, set_name=None, temporal_pooling='mean', flatten=True):
        self.meta_path = meta_path
        self.transform = transform
        self.set_name = set_name
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.task = task
        self.thresholds = thresholds

        self.samples = []
        self.source_counter = Counter()
        self.video_source_counter = Counter()
        self.video_id_to_source = dict()
        for fname, entry in meta.items():
            src = entry.get('source')
            p = entry.get('pain_level')
            vid = entry.get('video_id')
            if src not in source_filter or p is None:
                continue
            if (video_ids is not None) and (vid not in video_ids):
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
            self.source_counter[src] += 1
            if vid not in self.video_id_to_source:
                self.video_id_to_source[vid] = set()
            self.video_id_to_source[vid].add(src)
        # After collecting, compute video_source_counter
        for vid, sources in self.video_id_to_source.items():
            for src in sources:
                self.video_source_counter[src] += 1
        self.samples.sort(key=lambda x: x['fname'])
        if self.task == 'classification':
            invalid = [i for i, s in enumerate(self.samples) if not isinstance(s['class_label'], int)]
            if invalid:
                raise ValueError(f"Found non-integer class_label(s) at indices {invalid[:10]} (showing up to 10): "
                                 f"{[self.samples[i]['class_label'] for i in invalid[:10]]}")
        self._example_shape = None
        if self.samples:
            arr = np.load(os.path.join(os.path.dirname(meta_path), self.samples[0]['fname']))
            arr = torch.from_numpy(arr).float()
            # Apply pooling logic for true shape after processing
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
                pass  # Do not pool, keep temporal dimension
            if self.flatten:
                arr = arr.view(-1)
            self._example_shape = tuple(arr.shape)
        num_clips = len(self.samples)
        num_videos = len(set(x['video_id'] for x in self.samples))
        name_str = f" [{self.set_name}]" if self.set_name else ""
        # Get clip counts by source
        n_orig = self.source_counter.get("syracuse_original", 0)
        n_aug = self.source_counter.get("syracuse_aug", 0)
        # Video counts by source (video may be in both aug/orig)
        n_v_orig = self.video_source_counter.get("syracuse_original", 0)
        n_v_aug = self.video_source_counter.get("syracuse_aug", 0)
        print(f"[INFO] SyracuseDataset{name_str}: {num_clips} clips (original: {n_orig}, aug: {n_aug}); "
              f"{num_videos} videos (original: {n_v_orig}, aug: {n_v_aug})")
        if self.set_name == "VAL" and num_clips == 0:
            print("[WARN] SyracuseDataset [VAL] constructed with ZERO validation samples. Please check your val_vids and meta coherence.")

    def get_class_label(self, pain):
        """
        Maps pain score to class label based on threshold list for classification.
        """
        if self.task != 'classification' or self.thresholds is None:
            return None
        t = self.thresholds
        if len(t) == 0:
            return int(pain > 0)
        for i, th in enumerate(t):
            if pain <= th:
                return i
        return len(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        arr = np.load(os.path.join(os.path.dirname(self.meta_path), x['fname']))
        arr = torch.from_numpy(arr).float()
        # --- Temporal Pooling ---
        if self.temporal_pooling == 'mean':
            arr = arr.mean(dim=1) # [C,H,W]
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
            pass  # Do not pool, keep temporal dimension
        else:
            raise ValueError(f"Invalid temporal_pooling: {self.temporal_pooling}")
        if self.flatten:
            arr = arr.view(-1)
        if self.transform:
            arr = self.transform(arr)
        ret = [arr, x['pain_level']]
        if self.task == 'classification':
            cl = x['class_label']
            assert isinstance(cl, int), f"class_label not int in __getitem__: {cl} (pain_level={x['pain_level']})"
            ret.append(torch.tensor(cl, dtype=torch.long))
        ret += [x['video_id'], x['clip_id']]
        # print(f"sample: {x['fname']}, arr shape: {arr.shape}, label: {cl}")
        return tuple(ret)

    def label_distribution(self):
        return Counter([x['class_label'] for x in self.samples if x['class_label'] is not None])

    def __repr__(self):
        s = f'<SyracuseDataset: {len(self)} samples'
        if self._example_shape:
            s += f', sample shape={self._example_shape}'
        if self.samples and self.samples[0]['class_label'] is not None:
            s += f', class dist={dict(self.label_distribution())}'
        s += '>'
        return s

    @property
    def example_shape(self):
        """
        Shape of processed feature after pooling; e.g. (16, 4, 128, 128) for 'sample', (16, 128, 128) for 'mean'.
        """
        return self._example_shape

class SyracuseDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Syracuse dataset with pain level as ground truth and YAML-configurable classification.
    Provides cross-validation splits and optional balanced sampling for classification.
    Args:
        temporal_pooling: 'mean', 'max', 'sample', or 'none'.
            'sample' randomly picks 4 frames, pads if short.
            'none' keeps the temporal dimension (no pooling).
    """
    def __init__(self, meta_path, task: str = "classification", thresholds: list = None, batch_size=32, num_workers=4, cv_fold=3, seed=42,
                 balanced_sampling=False, transform=None, temporal_pooling='mean', flatten=True):
        super().__init__()
        self.meta_path = meta_path
        self.task = task
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.num_workers = num_workers
        # cv_fold specifies which split to use for validation:
        #   0, 1, 2: Use one fold for validation, remaining for training (3-fold CV)
        #   >=3: Special case, use all data for training (no validation set)
        self.cv_fold = cv_fold
        self.seed = seed
        self.balanced_sampling = balanced_sampling
        self.transform = transform
        self.temporal_pooling = temporal_pooling
        self.flatten = flatten
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the train/val datasets by stratified video CV split for classification,
        or by pseudo-stratified CV using uniform binning if regression. Augmentations assigned to train.
        """
        meta_dir = os.path.dirname(self.meta_path)
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        originals = [e for e in meta.values() if e.get('source') == 'syracuse_original' and e.get('pain_level') is not None]
        orig_video_ids = sorted({e['video_id'] for e in originals})
        pain_per_vid = {e['video_id']: e['pain_level'] for e in originals}

        if self.task == 'classification':
            if self.thresholds is None:
                raise ValueError("Thresholds must be provided for classification task in SyracuseDataModule.")
            current_thresholds = self.thresholds
            def label_fn(pain):
                for i, th in enumerate(current_thresholds):
                    if pain <= th:
                        return i
                return len(current_thresholds)
            split_vec = [label_fn(pain_per_vid[vid]) for vid in orig_video_ids]
            skf = StratifiedKFold(n_splits=3, random_state=self.seed, shuffle=True)
        elif self.task == 'regression':
            # Always use uniform binning from 0-10 for pseudo-stratification
            n_bins = 5
            bin_edges = np.linspace(0, 10, n_bins+1)[1:]  # [2,4,6,8,10]
            def pseudo_label(pain):
                for i, th in enumerate(bin_edges):
                    if pain <= th:
                        return i
                return n_bins - 1
            split_vec = [pseudo_label(pain_per_vid[vid]) for vid in orig_video_ids]
            skf = StratifiedKFold(n_splits=3, random_state=self.seed, shuffle=True)
            # Comment: These pseudo-labels are used only for splitting, not for prediction.
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

        train_vids, val_vids = None, None
        # Use one fold as val and others as train if cv_fold in [0,1,2].
        if self.cv_fold in [0, 1, 2]:
            for fold_idx, (train_index, val_index) in enumerate(skf.split(orig_video_ids, split_vec)):
                if fold_idx == self.cv_fold:
                    train_vids = set([orig_video_ids[i] for i in train_index])
                    val_vids   = set([orig_video_ids[i] for i in val_index])
                    break
        else:
            # For cv_fold >= 3, assign all data to training and leave validation empty.
            train_vids = set(orig_video_ids)
            val_vids = set()
        print(f"[DEBUG] train_vids: {len(train_vids)}; sample: {list(train_vids)[:5]}")
        print(f"[DEBUG] val_vids: {len(val_vids)}; sample: {list(val_vids)[:5]}")
        self.train_dataset = SyracuseDataset(
            self.meta_path,
            source_filter=["syracuse_original", "syracuse_aug"],
            video_ids=train_vids,
            task=self.task,
            thresholds=self.thresholds,
            transform=self.transform,
            set_name="TRAIN",
            temporal_pooling=self.temporal_pooling,
            flatten=self.flatten
        )
        if self.cv_fold < 3:
            self.val_dataset = SyracuseDataset(
                self.meta_path,
                source_filter=["syracuse_original"],
                video_ids=val_vids,
                task=self.task,
                thresholds=self.thresholds,
                transform=self.transform,
                set_name="VAL",
                temporal_pooling=self.temporal_pooling,
                flatten=self.flatten
            )
        else:
            self.val_dataset = None
        print(f"[INFO] Train class dist: {self.train_dataset.label_distribution()}")
        print(f"[INFO] Val class dist: {self.val_dataset.label_distribution()}")

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
        s = f"<SyracuseDataModule: batch_size={self.batch_size}, num_workers={self.num_workers}, cv_fold={self.cv_fold}, config={os.path.basename(self.meta_path)}>\n"
        if hasattr(self, 'train_dataset'):
            s += f"Train: {self.train_dataset}\n"
        if hasattr(self, 'val_dataset'):
            s += f"Val: {self.val_dataset}"
        return s

    @property
    def example_shape(self):
        """
        Shape of feature for model input, provided by train_dataset.
        """
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return self.train_dataset.example_shape
        else:
            return None

# Usage example:
# dm = SyracuseDataModule(meta_path="/home/nbi/marlin/wan_features/meta.json", task="classification", thresholds=[0, 2, 4, 6, 8], cv_fold=0, balanced_sampling=True)
# dm.setup()
# print(dm)

# dm = SyracuseDataModule(meta_path="/home/nbi/marlin/wan_features/meta.json", task="classification", thresholds=[0, 2, 4, 6, 8], cv_fold=0, debug_samples=100, balanced_sampling=True)
# dm.setup()
# loader = dm.train_dataloader()
# batch = next(iter(loader))
# for i, b in enumerate(batch):
#     print(f"Batch part {i}: dtype={getattr(b, 'dtype', type(b))}, shape={getattr(b, 'shape', '-')}, ex items={b[:5] if hasattr(b, '__getitem__') else b}") 