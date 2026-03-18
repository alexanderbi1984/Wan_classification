"""
Run a single LOSO (Leave-One-Subject-Out) fold for BioVid pain classification.

Called from a SLURM bash script that loops over test subjects:
    torchrun --nproc_per_node=4 scripts/train_loso_fold.py \
        --config config_pain/config_lora_t100_aug3.yaml \
        --test_subject 071309_w_21

Each fold:
    1. Assigns 1 subject as test, 5 cyclic neighbors as validation, rest as train
    2. Trains from scratch with early stopping
    3. Tests with best checkpoint
    4. Saves per-fold metrics to results/loso/{subject_id}.json
    5. Cleans up all checkpoints (only persists the metrics JSON)
"""

import argparse
import json
import os
import shutil
import sys
import time

import pandas as pd
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from classifier.online_multimodal import BioVidOnlineClassifier
from data.online_video import BioVidOnlineDataModule
from train_online_wan import build_model

ALL_SUBJECTS = [
    "071309_w_21", "071313_m_41", "071614_m_20", "071709_w_23", "071814_w_23",
    "071911_w_24", "072414_m_23", "072514_m_27", "072609_w_23", "072714_m_23",
    "073109_w_28", "073114_m_25", "080209_w_26", "080309_m_29", "080314_w_25",
    "080609_w_27", "080614_m_24", "080709_m_24", "080714_m_23", "081014_w_27",
    "081609_w_40", "081617_m_27", "081714_m_36", "082014_w_24", "082109_m_53",
    "082208_w_45", "082315_w_60", "082414_m_64", "082714_m_22", "082809_m_26",
    "082814_w_46", "082909_m_47", "083009_w_42", "083013_w_47", "083109_m_60",
    "083114_w_55", "091809_w_43", "091814_m_37", "091914_m_46", "092009_m_54",
    "092014_m_56", "092509_w_51", "092514_m_50", "092714_m_64", "092808_m_51",
    "092813_w_24", "100117_w_36", "100214_m_50", "100417_m_44", "100509_w_43",
    "100514_w_51", "100909_w_65", "100914_m_39", "101015_w_43", "101114_w_37",
    "101209_w_61", "101216_m_40", "101309_m_48", "101514_w_36", "101609_m_36",
    "101809_m_59", "101814_m_58", "101908_m_61", "101916_m_40", "102008_w_22",
    "102214_w_36", "102309_m_61", "102316_w_50", "102414_w_58", "102514_w_40",
    "110614_m_42", "110810_m_62", "110909_m_29", "111313_m_64", "111409_w_63",
    "111609_m_65", "111914_w_63", "112009_w_43", "112016_m_25", "112209_m_51",
    "112310_m_20", "112610_w_60", "112809_w_23", "112909_w_20", "112914_w_51",
    "120514_w_56", "120614_w_61",
]


def create_loso_csv(original_csv, test_subject, val_subjects, output_path):
    """Create a temporary CSV with modified splits for one LOSO fold."""
    df = pd.read_csv(original_csv)
    df["split"] = "train"
    df.loc[df["subject_id"] == test_subject, "split"] = "test"
    for vs in val_subjects:
        df.loc[df["subject_id"] == vs, "split"] = "val"
    df.to_csv(output_path, index=False)

    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    print(f"[LOSO] test={test_subject} | train={n_train}, val={n_val}, test={n_test}")


def main():
    parser = argparse.ArgumentParser(description="LOSO fold training for BioVid")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_subject", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/loso")
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_val_subjects", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config (e.g. 20 to reduce OOM risk)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Check if fold already completed (resume-safe)
    result_file = os.path.join(args.output_dir, f"{args.test_subject}.json")
    if os.path.exists(result_file):
        print(f"[LOSO] Fold {args.test_subject} already completed, skipping.")
        return

    # Determine cyclic validation subjects
    idx = ALL_SUBJECTS.index(args.test_subject)
    val_subjects = [
        ALL_SUBJECTS[(idx + j) % len(ALL_SUBJECTS)]
        for j in range(1, args.n_val_subjects + 1)
    ]

    # Temp file paths
    tmp_csv = f"/tmp/loso_{args.test_subject}.csv"
    ready_flag = f"/tmp/loso_{args.test_subject}.ready"
    ckpt_dir = f"/tmp/loso_ckpt_{args.test_subject}"

    # Rank 0 creates temp CSV; others wait
    if local_rank == 0:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        create_loso_csv(config["data"]["labels_csv"], args.test_subject, val_subjects, tmp_csv)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        open(ready_flag, "w").close()
    else:
        while not os.path.exists(ready_flag):
            time.sleep(0.1)
        time.sleep(0.5)
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Override CSV path
    config["data"]["labels_csv"] = tmp_csv
    pl.seed_everything(args.seed)

    # Build model
    model = build_model(config)

    # Build data module
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    aug_cfg = config.get("augmentation", None)

    batch_size = args.batch_size if args.batch_size else train_cfg.get("batch_size", 24)

    dm = BioVidOnlineDataModule(
        labels_csv=tmp_csv,
        frames_root=data_cfg["frames_root"],
        num_classes=data_cfg.get("num_classes", 5),
        batch_size=batch_size,
        num_workers=train_cfg.get("num_workers", 8),
        resize=data_cfg.get("resize", 128),
        max_frames=data_cfg.get("max_frames", 129),
        sample_rate=data_cfg.get("sample_rate", 1),
        augmentation=aug_cfg,
    )
    dm.setup()

    # Callbacks — save checkpoints to /tmp, no logging to save disk
    monitor = train_cfg.get("monitor_metric", "val_pain_QWK")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=False,
        filename="loso_best",
        monitor=monitor,
        mode="max" if "QWK" in monitor else "min",
        save_top_k=1,
    )
    es_cb = EarlyStopping(
        monitor=monitor,
        mode="max" if "QWK" in monitor else "min",
        patience=train_cfg.get("early_stop_patience", 10),
        verbose=True,
    )

    strategy = DDPStrategy(find_unused_parameters=True) if args.n_gpus > 1 else "auto"

    trainer = Trainer(
        log_every_n_steps=10,
        devices=args.n_gpus,
        accelerator="gpu",
        strategy=strategy,
        max_epochs=train_cfg.get("max_epochs", 200),
        precision=train_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        callbacks=[ckpt_cb, es_cb],
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic="warn",
        num_sanity_val_steps=0,
        logger=False,
        benchmark=True,
    )

    # --- Train ---
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    best_path = ckpt_cb.best_model_path
    best_score = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None
    print(f"[LOSO] {args.test_subject}: best checkpoint = {best_path}, score = {best_score}")

    # --- Test ---
    test_results = trainer.test(
        model=None,
        dataloaders=dm.test_dataloader(),
        ckpt_path=best_path,
    )

    # Save results (rank 0 only)
    if local_rank == 0:
        metrics = test_results[0] if test_results else {}

        # Count true samples per class in this fold's test set
        test_df = pd.read_csv(tmp_csv)
        test_df = test_df[test_df["split"] == "test"]
        true_counts = test_df["pain_level"].value_counts().sort_index().to_dict()

        result = {
            "test_subject": args.test_subject,
            "val_subjects": val_subjects,
            "best_epoch": best_path.split("=")[-1].replace(".ckpt", "") if best_path else None,
            "best_val_score": best_score,
            "true_counts_per_class": true_counts,
            "n_test_samples": len(test_df),
            "metrics": metrics,
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[LOSO] Results saved to {result_file}")

        # Clean up temp files
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        for fp in [tmp_csv, ready_flag]:
            try:
                os.unlink(fp)
            except OSError:
                pass

    print(f"[LOSO] Fold {args.test_subject} complete.")


if __name__ == "__main__":
    main()
