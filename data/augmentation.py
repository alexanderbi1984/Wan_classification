"""Video clip augmentation for RGB-based pain classification.

Adapted from MMA project's ``clip_augment.py`` with added RGB-specific
augmentations (horizontal flip, color jitter, random erasing) that were
not applicable in MMA's 5-to-3 multimodal fusion setting but are effective
for our raw RGB pipeline.

All spatial/color transforms are applied *consistently* across frames
within a clip to preserve temporal coherence.

Input format: ``(C, T, H, W)`` float32 in ``[0, 1]``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF

try:
    from torchvision.transforms import AugMix, TrivialAugmentWide
    _HAS_ADVANCED_AUG = True
except ImportError:
    AugMix = None
    TrivialAugmentWide = None
    _HAS_ADVANCED_AUG = False


@dataclass
class VideoAugmentConfig:
    """Configuration for :class:`VideoClipAugmentor`.

    Basic RGB augmentations (applied with per-clip consistency):
        hflip_prob: Probability of random horizontal flip.
        jitter_prob: Probability of applying color jitter.
        jitter_brightness / contrast / saturation / hue: ColorJitter ranges.
        grayscale_prob: Probability of converting to grayscale.
        erasing_prob: Probability of random erasing (cutout).
        erasing_scale: Area fraction range for random erasing.

    Advanced augmentations (AugMix / TrivialAugmentWide, from MMA):
        advanced_aug_prob: Probability of applying AugMix or TrivialAugment.
        augmix_weight: Weight for AugMix vs TrivialAugment (0 = TrivialAug only).
        augmix_severity: AugMix severity level.
    """
    hflip_prob: float = 0.5
    jitter_prob: float = 0.3
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
    grayscale_prob: float = 0.1
    erasing_prob: float = 0.2
    erasing_scale: tuple = (0.02, 0.2)
    advanced_aug_prob: float = 0.0
    augmix_weight: float = 0.0
    augmix_severity: int = 2


class VideoClipAugmentor:
    """Apply augmentations to an RGB video clip ``(C, T, H, W)``.

    All transforms use a single random seed per clip so every frame
    sees the identical spatial/color transformation, preserving
    temporal coherence.
    """

    def __init__(self, config: VideoAugmentConfig) -> None:
        self.config = config
        self._rng = random.Random()

        if config.advanced_aug_prob > 0.0 and _HAS_ADVANCED_AUG:
            try:
                self._augmix = AugMix(
                    severity=config.augmix_severity, mixture_width=3,
                    chain_depth=-1, alpha=1.0, all_ops=True,
                )
            except TypeError:
                self._augmix = AugMix(severity=config.augmix_severity)
            self._trivial = TrivialAugmentWide(num_magnitude_bins=31)
        else:
            self._augmix = None
            self._trivial = None

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Augment a video clip in-place-safe manner.

        Args:
            clip: ``(C, T, H, W)`` float32 tensor in ``[0, 1]``.

        Returns:
            Augmented clip, same shape and dtype.
        """
        if clip.numel() == 0:
            return clip

        C, T, H, W = clip.shape
        out = clip.clone()

        # --- Random horizontal flip (consistent across frames) ---
        if self._rng.random() < self.config.hflip_prob:
            out = out.flip(-1)

        # --- Color jitter (consistent across frames) ---
        if self._rng.random() < self.config.jitter_prob:
            out = self._apply_jitter(out)

        # --- Random grayscale (consistent across frames) ---
        if self._rng.random() < self.config.grayscale_prob:
            out = self._apply_grayscale(out)

        # --- Advanced augmentation (AugMix / TrivialAugment) ---
        if self._rng.random() < self.config.advanced_aug_prob:
            out = self._apply_advanced_aug(out)

        # --- Random erasing (consistent region across frames) ---
        if self._rng.random() < self.config.erasing_prob:
            out = self._apply_erasing(out)

        return out.clamp_(0.0, 1.0)

    def _apply_jitter(self, clip: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        brightness = 1.0 + self._rng.uniform(-cfg.jitter_brightness, cfg.jitter_brightness)
        contrast = 1.0 + self._rng.uniform(-cfg.jitter_contrast, cfg.jitter_contrast)
        saturation = 1.0 + self._rng.uniform(-cfg.jitter_saturation, cfg.jitter_saturation)
        hue = self._rng.uniform(-cfg.jitter_hue, cfg.jitter_hue)

        C, T, H, W = clip.shape
        for t in range(T):
            frame = clip[:, t]  # (C, H, W)
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            frame = TF.adjust_saturation(frame, saturation)
            frame = TF.adjust_hue(frame, hue)
            clip[:, t] = frame
        return clip

    def _apply_grayscale(self, clip: torch.Tensor) -> torch.Tensor:
        C, T, H, W = clip.shape
        for t in range(T):
            clip[:, t] = TF.rgb_to_grayscale(clip[:, t].unsqueeze(0), num_output_channels=3).squeeze(0)
        return clip

    def _apply_erasing(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply the same random erasing rectangle to all frames."""
        C, T, H, W = clip.shape
        lo, hi = self.config.erasing_scale
        area = H * W
        erase_area = self._rng.uniform(lo, hi) * area
        aspect = self._rng.uniform(0.3, 3.3)

        eh = int(round((erase_area * aspect) ** 0.5))
        ew = int(round((erase_area / aspect) ** 0.5))
        eh = min(eh, H)
        ew = min(ew, W)

        top = self._rng.randint(0, H - eh)
        left = self._rng.randint(0, W - ew)

        clip[:, :, top:top + eh, left:left + ew] = 0.0
        return clip

    def _apply_advanced_aug(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply AugMix or TrivialAugmentWide (from MMA's clip_augment)."""
        if self._augmix is None and self._trivial is None:
            return clip

        if self.config.augmix_weight > 0 and self._augmix is not None:
            if self._rng.random() < self.config.augmix_weight:
                transform = self._augmix
            else:
                transform = self._trivial
        else:
            transform = self._trivial

        if transform is None:
            return clip

        C, T, H, W = clip.shape
        base_seed = self._rng.randint(0, 2**31 - 1)

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        try:
            for t in range(T):
                frame = clip[:, t]
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(base_seed)
                    random.seed(base_seed)
                    np.random.seed(base_seed % (2**32 - 1))

                    frame_uint8 = (frame * 255.0).round().clamp_(0, 255).to(torch.uint8)
                    try:
                        aug = transform(frame_uint8)
                    except Exception:
                        aug = frame_uint8

                    if aug.dtype == torch.uint8:
                        aug = aug.float().div_(255.0)
                    clip[:, t] = aug.clamp_(0.0, 1.0)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

        return clip


def build_augmentor(aug_cfg: Optional[dict] = None) -> Optional[VideoClipAugmentor]:
    """Build a VideoClipAugmentor from a config dict.

    Returns None if aug_cfg is None or all probabilities are zero.
    """
    if aug_cfg is None:
        return None

    config = VideoAugmentConfig(
        hflip_prob=aug_cfg.get("hflip_prob", 0.5),
        jitter_prob=aug_cfg.get("jitter_prob", 0.3),
        jitter_brightness=aug_cfg.get("jitter_brightness", 0.2),
        jitter_contrast=aug_cfg.get("jitter_contrast", 0.2),
        jitter_saturation=aug_cfg.get("jitter_saturation", 0.2),
        jitter_hue=aug_cfg.get("jitter_hue", 0.05),
        grayscale_prob=aug_cfg.get("grayscale_prob", 0.1),
        erasing_prob=aug_cfg.get("erasing_prob", 0.2),
        erasing_scale=tuple(aug_cfg.get("erasing_scale", [0.02, 0.2])),
        advanced_aug_prob=aug_cfg.get("advanced_aug_prob", 0.0),
        augmix_weight=aug_cfg.get("augmix_weight", 0.0),
        augmix_severity=aug_cfg.get("augmix_severity", 2),
    )

    has_any = (config.hflip_prob > 0 or config.jitter_prob > 0
               or config.grayscale_prob > 0 or config.erasing_prob > 0
               or config.advanced_aug_prob > 0)
    if not has_any:
        return None

    return VideoClipAugmentor(config)
