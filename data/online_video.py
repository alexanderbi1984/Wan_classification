"""
Online video dataset and data module for end-to-end training with LoRA.

Instead of loading pre-extracted .npy features, this module loads raw video
frames and returns tensors suitable for the WanVAE + WanModel pipeline.

Supports:
    - BioVid pain classification from frame directories + CSV labels
      (same dataset as MMA project)
    - Generic video file loading (mp4, etc.)
"""

import csv
import json
import os
import random
from collections import Counter
from typing import List, Optional, Set

import cv2
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from data.augmentation import VideoClipAugmentor, build_augmentor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def read_frames_from_directory(
    frame_dir: str,
    max_frames: int = 129,
    resize: int = 128,
    sample_rate: int = 1,
) -> List[torch.Tensor]:
    """Read image frames from a directory of individual frame files.

    Matches the BioVid frame layout: {video_id}/frame_det_00_000001.bmp, ...

    Args:
        frame_dir: Directory containing frame image files.
        max_frames: Maximum number of frames to load.
        resize: Resize frames to (resize, resize).
        sample_rate: Take every Nth frame (1 = all frames).

    Returns:
        List of frame tensors (C, H, W), float32, normalized to [0, 1].
    """
    if not os.path.isdir(frame_dir):
        raise IOError(f"Frame directory not found: {frame_dir}")

    all_files = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not all_files:
        raise IOError(f"No image files found in {frame_dir}")

    selected = all_files[::sample_rate][:max_frames]

    frames = []
    for fname in selected:
        path = os.path.join(frame_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (resize, resize))
        tensor = torch.from_numpy(img).float() / 255.0  # (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # (C, H, W)
        frames.append(tensor)

    return frames


def read_video_frames(
    video_path: str,
    max_frames: int = 129,
    resize: int = 128,
    sample_fps: int = 3,
) -> List[torch.Tensor]:
    """Read and sample frames from a video file (.mp4, .avi, etc.).

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.
        resize: Resize frames to (resize, resize).
        sample_fps: Target sampling frame rate.

    Returns:
        List of frame tensors (C, H, W), float32, normalized to [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0
    sample_interval = max(int(input_fps / sample_fps), 1)

    frames, count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            tensor = torch.from_numpy(frame).float() / 255.0
            tensor = tensor.permute(2, 0, 1)
            frames.append(tensor)
            if len(frames) >= max_frames:
                break
        count += 1
    cap.release()
    return frames


# ------------------------------------------------------------------ #
#  BioVid Dataset (frame directories + CSV — same format as MMA)     #
# ------------------------------------------------------------------ #

class BioVidOnlineDataset(Dataset):
    """Dataset for BioVid pain classification using frame directories.

    Loads frames from {frames_root}/{video_id}/*.bmp and labels from a CSV
    file (same format as MMA project's biovid_pain_labels.csv).

    Each __getitem__ returns:
        video_tensor: (C, T, H, W) float32, normalized to [0, 1]
        label: int (pain level 0-4)

    Args:
        labels_csv: Path to CSV with columns: video_id, pain_level, split, ...
        frames_root: Root directory containing per-video frame directories.
        split: Which split to load ('train', 'val', 'test', or None for all).
        num_classes: Number of pain classes (default 5).
        resize: Spatial resolution for frames.
        max_frames: Maximum number of frames to load per video.
        sample_rate: Take every Nth frame from the sorted frame list.
    """

    def __init__(
        self,
        labels_csv: str,
        frames_root: str,
        split: Optional[str] = None,
        num_classes: int = 5,
        resize: int = 128,
        max_frames: int = 129,
        sample_rate: int = 1,
        augmentor: Optional[VideoClipAugmentor] = None,
        class_subset: Optional[List[int]] = None,
    ):
        self.frames_root = frames_root
        self.resize = resize
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.augmentor = augmentor

        # Label remapping for class_subset (e.g., [0, 4] → {0:0, 4:1})
        self._label_map = None
        if class_subset is not None:
            class_subset = sorted(class_subset)
            self._label_map = {orig: new for new, orig in enumerate(class_subset)}

        df = pd.read_csv(labels_csv)

        if split is not None and "split" in df.columns:
            df = df[df["split"].str.lower() == split.lower()].copy()

        if class_subset is not None:
            df = df[df["pain_level"].isin(class_subset)].copy()

        self.samples = []
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            label = int(row["pain_level"])
            if self._label_map is not None:
                label = self._label_map[label]
            frame_dir = os.path.join(frames_root, video_id)
            if os.path.isdir(frame_dir):
                self.samples.append({"video_id": video_id, "label": label})

        self.samples.sort(key=lambda x: x["video_id"])
        split_str = split or "all"
        subset_str = f" classes={class_subset}" if class_subset else ""
        print(f"[BioVidOnlineDataset] [{split_str}]{subset_str} {len(self.samples)} videos, "
              f"label dist: {dict(self.label_distribution())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_dir = os.path.join(self.frames_root, sample["video_id"])

        frames = read_frames_from_directory(
            frame_dir,
            max_frames=self.max_frames,
            resize=self.resize,
            sample_rate=self.sample_rate,
        )

        if not frames:
            raise RuntimeError(f"No frames loaded from {frame_dir}")

        video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)

        if self.augmentor is not None:
            video_tensor = self.augmentor(video_tensor)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return video_tensor, label

    def label_distribution(self):
        return Counter(s["label"] for s in self.samples)


def collate_biovid_online(batch):
    """Collate function for BioVid that pads variable-length videos.

    Returns:
        videos: (B, C, T_max, H, W)
        labels: (B,)
    """
    videos, labels = zip(*batch)

    max_t = max(v.shape[1] for v in videos)
    C, _, H, W = videos[0].shape

    padded = []
    for v in videos:
        t = v.shape[1]
        if t < max_t:
            pad = torch.zeros(C, max_t - t, H, W, dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        padded.append(v)

    return torch.stack(padded), torch.stack(list(labels))


class BioVidOnlineDataModule(pl.LightningDataModule):
    """Data module for BioVid online video training.

    Uses the same CSV format and frame directory layout as the MMA project.

    Args:
        labels_csv: Path to biovid_pain_labels.csv.
        frames_root: Path to biovid_cropped_frames/.
        num_classes: Number of ordinal pain classes.
        batch_size: Batch size for data loaders.
        num_workers: Number of data loader workers.
        resize: Video resize resolution.
        max_frames: Maximum frames per video.
        sample_rate: Take every Nth frame.
    """

    def __init__(
        self,
        labels_csv: str,
        frames_root: str,
        num_classes: int = 5,
        batch_size: int = 1,
        num_workers: int = 4,
        resize: int = 128,
        max_frames: int = 129,
        sample_rate: int = 1,
        augmentation: Optional[dict] = None,
        class_subset: Optional[List[int]] = None,
    ):
        super().__init__()
        self.labels_csv = labels_csv
        self.frames_root = frames_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.augmentation = augmentation
        self.class_subset = class_subset

    def setup(self, stage=None):
        train_augmentor = build_augmentor(self.augmentation)
        if train_augmentor is not None:
            print(f"[BioVidOnlineDataModule] Train augmentation enabled: "
                  f"hflip={self.augmentation.get('hflip_prob', 0.5)}, "
                  f"jitter={self.augmentation.get('jitter_prob', 0.3)}, "
                  f"grayscale={self.augmentation.get('grayscale_prob', 0.1)}, "
                  f"erasing={self.augmentation.get('erasing_prob', 0.2)}")

        common_kwargs = dict(
            labels_csv=self.labels_csv,
            frames_root=self.frames_root,
            num_classes=self.num_classes,
            resize=self.resize,
            max_frames=self.max_frames,
            sample_rate=self.sample_rate,
            class_subset=self.class_subset,
        )
        self._train_ds = BioVidOnlineDataset(split="train", augmentor=train_augmentor, **common_kwargs)
        self._val_ds = BioVidOnlineDataset(split="val", **common_kwargs)
        self._test_ds = BioVidOnlineDataset(split="test", **common_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_biovid_online,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not self._val_ds or len(self._val_ds) == 0:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_biovid_online,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not self._test_ds or len(self._test_ds) == 0:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_biovid_online,
            pin_memory=True,
        )
