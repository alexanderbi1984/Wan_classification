"""
Ensemble evaluation for BioVid pain classification.

Loads multiple trained checkpoints, runs each on the test set, and
averages their CORAL sigmoid probabilities before decoding.

Usage:
    torchrun --standalone --nproc_per_node=4 scripts/test_ensemble.py \
        --config config_pain/config_lora_t100_aug3.yaml \
        --ckpts ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from classifier.online_multimodal import BioVidOnlineClassifier, OnlineMultimodalClassifier
from data.online_video import BioVidOnlineDataModule


class EnsembleModel(pl.LightningModule):
    """Sequential ensemble: loads one model at a time to minimize VRAM.

    Instead of holding all models in memory simultaneously (which needs
    N × 30 GB for Wan 14B), this collects per-sample probabilities from
    each model sequentially during the first pass, then computes metrics
    from the averaged probabilities.
    """

    def __init__(self, ckpt_paths: list, num_classes: int = 5, eval_head: str = "coral"):
        super().__init__()
        self.ckpt_paths = ckpt_paths
        self.n_cls = num_classes
        self.eval_head = eval_head
        self._all_probs = []   # list of lists, one per model
        self._all_labels = []

        import torchmetrics
        from torchmetrics.classification import (
            CohenKappa, Accuracy, MulticlassF1Score,
            MulticlassRecall, MulticlassConfusionMatrix,
        )
        ma = {"dist_sync_on_step": False}
        self.test_pain_mae = torchmetrics.MeanAbsoluteError(**ma)
        self.test_pain_qwk = CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic", **ma)
        self.test_pain_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro", **ma)
        self.test_pain_f1 = MulticlassF1Score(num_classes=num_classes, average="macro", **ma)
        self.test_pain_recall = MulticlassRecall(num_classes=num_classes, average="none", **ma)
        self.test_pain_cm = MulticlassConfusionMatrix(num_classes=num_classes, **ma)

    def run_sequential_ensemble(self, dataloader, device):
        """Run each model one at a time, collect probabilities."""
        use_ce = self.eval_head == "ce"
        all_model_probs = []

        for i, ckpt_path in enumerate(self.ckpt_paths):
            print(f"\n[Ensemble] Loading model {i+1}/{len(self.ckpt_paths)}: {ckpt_path}")
            try:
                model = BioVidOnlineClassifier.load_from_checkpoint(ckpt_path)
            except TypeError as e:
                # Old checkpoints may have extra saved hparams (e.g. use_text_context)
                print(f"[Ensemble] Direct load failed ({e}), retrying with filtered hparams...")
                ckpt = torch.load(ckpt_path, map_location="cpu")
                import inspect
                valid_params = set(inspect.signature(OnlineMultimodalClassifier.__init__).parameters.keys())
                filtered = {k: v for k, v in ckpt["hyper_parameters"].items() if k in valid_params}
                model = BioVidOnlineClassifier(**filtered)
                model.load_state_dict(ckpt["state_dict"], strict=False)
            model.eval()
            model.to(device)

            model_probs = []
            model_labels = []
            with torch.no_grad():
                for batch in dataloader:
                    video, labels = batch
                    video = video.to(device)
                    out = model(video)
                    if use_ce:
                        probs = torch.softmax(out["pain_ce"], dim=1)
                    else:
                        probs = torch.sigmoid(out["pain_coral"])
                    model_probs.append(probs.cpu())
                    model_labels.append(labels.cpu())

            all_model_probs.append(torch.cat(model_probs, dim=0))
            if i == 0:
                self._all_labels = torch.cat(model_labels, dim=0)

            # Free memory before loading next model
            del model
            torch.cuda.empty_cache()
            print(f"[Ensemble] Model {i+1} done, VRAM freed")

        # Average probabilities across models
        avg_probs = torch.stack(all_model_probs, dim=0).mean(dim=0)

        if use_ce:
            preds = avg_probs.argmax(dim=1)
        else:
            preds = OnlineMultimodalClassifier.prob_to_label(avg_probs, self.n_cls)

        labels = self._all_labels
        self.test_pain_mae.update(preds, labels)
        self.test_pain_qwk.update(preds, labels)
        self.test_pain_acc.update(preds, labels)
        self.test_pain_f1.update(preds, labels)
        self.test_pain_recall.update(preds, labels)
        self.test_pain_cm.update(preds, labels)

        self._print_results()

    def _print_results(self):
        qwk = self.test_pain_qwk.compute()
        mae = self.test_pain_mae.compute()
        acc = self.test_pain_acc.compute()
        f1 = self.test_pain_f1.compute()
        recall = self.test_pain_recall.compute()
        cm = self.test_pain_cm.compute()
        pred_counts = cm.sum(dim=0).long()
        true_counts = cm.sum(dim=1).long()

        print(f"\n{'='*60}")
        print(f"  Ensemble Results ({len(self.ckpt_paths)} models)")
        print(f"{'='*60}")
        print(f"  test_pain_QWK:      {qwk:.4f}")
        print(f"  test_pain_MAE:      {mae:.4f}")
        print(f"  test_pain_Accuracy: {acc:.4f}")
        print(f"  test_pain_F1:       {f1:.4f}")

        print(f"\n  Per-class recall: "
              + " | ".join(f"c{i}={recall[i]:.3f}" for i in range(self.n_cls)))
        print(f"  Pred distribution: "
              + " | ".join(f"c{i}={pred_counts[i]}" for i in range(self.n_cls))
              + f"  (true: {' | '.join(f'c{i}={true_counts[i]}' for i in range(self.n_cls))})")

        class_names = [f"c{i}" for i in range(self.n_cls)]
        header = "        " + "  ".join(f"{c:>6s}" for c in class_names)
        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        print(header)
        for i in range(self.n_cls):
            row = "  ".join(f"{int(cm[i, j]):6d}" for j in range(self.n_cls))
            print(f"  {class_names[i]:>4s}  {row}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation for BioVid")
    parser.add_argument("--config", type=str, required=True,
                        help="Config for data loading (any experiment config works)")
    parser.add_argument("--ckpts", type=str, nargs="+", required=True,
                        help="Paths to checkpoint files to ensemble")
    parser.add_argument("--eval_head", type=str, default="coral",
                        choices=["coral", "ce"])
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (use smaller for low-VRAM GPUs)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42)

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
        augmentation=None,
    )
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {torch.cuda.get_device_name(0)} — {total_gb:.1f} GB VRAM")

    print(f"[Ensemble] {len(args.ckpts)} models, batch_size={batch_size}")
    for i, c in enumerate(args.ckpts):
        print(f"  {i+1}: {c}")

    ensemble = EnsembleModel(
        ckpt_paths=args.ckpts,
        num_classes=data_cfg.get("num_classes", 5),
        eval_head=args.eval_head,
    )

    # Sequential single-GPU inference (loads one model at a time)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        ensemble.run_sequential_ensemble(dm.test_dataloader(), device)


if __name__ == "__main__":
    main()
