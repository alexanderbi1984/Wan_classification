"""
Test-Time Augmentation (TTA) for BioVid pain classification.

Runs N augmented forward passes per sample, averages CORAL sigmoid probabilities,
then decodes to ordinal labels. No retraining required.

Usage:
    torchrun --standalone --nproc_per_node=4 scripts/test_tta.py \
        --config config_pain/config_lora_t100_aug3.yaml \
        --ckpt_path results/.../checkpoints/best.ckpt \
        --n_aug 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from classifier.online_multimodal import BioVidOnlineClassifier, OnlineMultimodalClassifier
from data.online_video import BioVidOnlineDataModule
from data.augmentation import VideoAugmentConfig, VideoClipAugmentor


class TTAWrapper(pl.LightningModule):
    """Wraps a trained classifier to apply TTA at test time.

    Strategy: run N augmented forward passes, average the sigmoid (CORAL)
    or softmax (CE) probabilities, then decode predictions.
    """

    def __init__(self, model: OnlineMultimodalClassifier, augmentor, n_aug: int = 10):
        super().__init__()
        self.model = model
        self.augmentor = augmentor
        self.n_aug = n_aug
        self.model.eval()

        # Reuse metrics from the inner model
        import torchmetrics
        from torchmetrics.classification import (
            CohenKappa, Accuracy, MulticlassF1Score,
            MulticlassRecall, MulticlassConfusionMatrix,
        )
        n_cls = model.hparams.num_pain_classes
        ma = {"dist_sync_on_step": False}
        self.test_pain_mae = torchmetrics.MeanAbsoluteError(**ma)
        self.test_pain_qwk = CohenKappa(task="multiclass", num_classes=n_cls, weights="quadratic", **ma)
        self.test_pain_acc = Accuracy(task="multiclass", num_classes=n_cls, average="macro", **ma)
        self.test_pain_f1 = MulticlassF1Score(num_classes=n_cls, average="macro", **ma)
        self.test_pain_recall = MulticlassRecall(num_classes=n_cls, average="none", **ma)
        self.test_pain_cm = MulticlassConfusionMatrix(num_classes=n_cls, **ma)
        self.n_cls = n_cls

    def _apply_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch of videos (B, C, T, H, W) on GPU."""
        augmented = []
        for i in range(video.shape[0]):
            # video[i]: (C, T, H, W), values in [0, 1]
            clip = video[i].cpu()  # (C, T, H, W) float32
            aug_clip = self.augmentor(clip)  # (C, T, H, W) float32
            augmented.append(aug_clip.to(video.device))
        return torch.stack(augmented)

    def test_step(self, batch, batch_idx):
        video, labels = batch
        eval_head = self.model.hparams.get("eval_head", "coral")
        use_ce = eval_head == "ce"

        # Accumulate probabilities across augmentations
        avg_probs = None

        for aug_idx in range(self.n_aug):
            if aug_idx == 0:
                aug_video = video
            else:
                aug_video = self._apply_augmentation(video)

            with torch.no_grad():
                out = self.model(aug_video)

            if use_ce:
                probs = torch.softmax(out["pain_ce"], dim=1)  # (B, K)
            else:
                probs = torch.sigmoid(out["pain_coral"])  # (B, K-1)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs = avg_probs + probs

        avg_probs = avg_probs / self.n_aug

        if use_ce:
            preds = avg_probs.argmax(dim=1)
        else:
            preds = OnlineMultimodalClassifier.prob_to_label(avg_probs, self.n_cls)

        self.test_pain_mae.update(preds, labels)
        self.test_pain_qwk.update(preds, labels)
        self.test_pain_acc.update(preds, labels)
        self.test_pain_f1.update(preds, labels)
        self.test_pain_recall.update(preds, labels)
        self.test_pain_cm.update(preds, labels)

    def on_test_epoch_end(self):
        sd = {"sync_dist": True}
        self.log("test_pain_QWK", self.test_pain_qwk, **sd)
        self.log("test_pain_MAE", self.test_pain_mae, **sd)
        self.log("test_pain_Accuracy", self.test_pain_acc, **sd)
        self.log("test_pain_F1", self.test_pain_f1, **sd)

        recall = self.test_pain_recall.compute()
        cm = self.test_pain_cm.compute()
        pred_counts = cm.sum(dim=0).long()
        true_counts = cm.sum(dim=1).long()

        for i in range(self.n_cls):
            self.log(f"test_pain_recall_c{i}", float(recall[i]), sync_dist=True)
            self.log(f"test_pain_pred_count_c{i}", float(pred_counts[i]), sync_dist=True)

        print(f"[TTA Test] n_aug={self.n_aug}")
        print(f"[TTA Test] per-class recall: "
              + " | ".join(f"c{i}={recall[i]:.3f}" for i in range(self.n_cls)))
        print(f"[TTA Test] pred distribution: "
              + " | ".join(f"c{i}={pred_counts[i]}" for i in range(self.n_cls))
              + f"  (true: {' | '.join(f'c{i}={true_counts[i]}' for i in range(self.n_cls))})")

        class_names = [f"c{i}" for i in range(self.n_cls)]
        header = "        " + "  ".join(f"{c:>6s}" for c in class_names)
        print(f"\n[TTA Test] Confusion Matrix (rows=true, cols=pred):")
        print(header)
        for i in range(self.n_cls):
            row = "  ".join(f"{int(cm[i, j]):6d}" for j in range(self.n_cls))
            print(f"  {class_names[i]:>4s}  {row}")


def main():
    parser = argparse.ArgumentParser(description="TTA evaluation for BioVid")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_aug", type=int, default=10,
                        help="Number of augmented forward passes (including original)")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (use smaller for low-VRAM GPUs)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42)

    # Build data module (test set only)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    batch_size = args.batch_size or train_cfg.get("batch_size", 24)
    dm = BioVidOnlineDataModule(
        labels_csv=data_cfg["labels_csv"],
        frames_root=data_cfg["frames_root"],
        num_classes=data_cfg.get("num_classes", 5),
        batch_size=batch_size,
        num_workers=train_cfg.get("num_workers", 8),
        resize=data_cfg.get("resize", 128),
        max_frames=data_cfg.get("max_frames", 129),
        sample_rate=data_cfg.get("sample_rate", 1),
        augmentation=None,  # No augmentation on data loader side
    )
    dm.setup()

    # Build augmentor for TTA
    aug_cfg = config.get("augmentation", {})
    aug_config = VideoAugmentConfig(
        hflip_prob=aug_cfg.get("hflip_prob", 0.5),
        jitter_prob=aug_cfg.get("jitter_prob", 0.5),
        jitter_brightness=aug_cfg.get("jitter_brightness", 0.4),
        jitter_contrast=aug_cfg.get("jitter_contrast", 0.4),
        jitter_saturation=aug_cfg.get("jitter_saturation", 0.3),
        jitter_hue=aug_cfg.get("jitter_hue", 0.1),
        grayscale_prob=aug_cfg.get("grayscale_prob", 0.2),
        erasing_prob=aug_cfg.get("erasing_prob", 0.3),
        erasing_scale=tuple(aug_cfg.get("erasing_scale", [0.02, 0.33])),
        advanced_aug_prob=aug_cfg.get("advanced_aug_prob", 0.0),
        augmix_weight=aug_cfg.get("augmix_weight", 0.0),
        augmix_severity=aug_cfg.get("augmix_severity", 3),
    )
    augmentor = VideoClipAugmentor(aug_config)

    # Load model from checkpoint
    model = BioVidOnlineClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()

    tta_model = TTAWrapper(model, augmentor, n_aug=args.n_aug)

    strategy = "auto"
    if args.n_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = Trainer(
        devices=args.n_gpus,
        accelerator="gpu",
        strategy=strategy,
        precision=train_cfg.get("precision", "bf16-mixed"),
        enable_progress_bar=True,
        logger=False,
    )

    results = trainer.test(tta_model, dataloaders=dm.test_dataloader())
    print("\n=== TTA Results ===")
    for k, v in results[0].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
