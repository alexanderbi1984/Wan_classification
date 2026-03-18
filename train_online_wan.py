"""
Training script for online LoRA fine-tuning of the Wan DiT backbone.

Replaces pre-extracted features with end-to-end training:
    Raw video -> WanVAE (frozen) -> WanModel/DiT (LoRA) -> downstream classifier

Uses BioVid pain classification dataset (same as MMA project).

Usage:
    python train_online_wan.py --config config_pain/config_lora.yaml
    python train_online_wan.py --config config_pain/config_lora.yaml --mode test --ckpt_path path/to/ckpt
"""

import argparse
import json
import os
from datetime import datetime

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from classifier.online_multimodal import BioVidOnlineClassifier
from data.online_video import BioVidOnlineDataModule


def get_output_dir(run_name, base_dir="results"):
    """Create timestamped output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(base_dir, f"{run_name}_{timestamp}")
    logs = os.path.join(base, "logs")
    ckpt = os.path.join(base, "checkpoints")
    results = os.path.join(base, "results")

    for d in (logs, ckpt, results):
        os.makedirs(d, exist_ok=True)
    return base, logs, ckpt, results, timestamp


def save_hparams_to_dir(hparams, out_dir, timestamp):
    """Persist hyperparameters as both YAML and JSON."""
    with open(os.path.join(out_dir, f"hparams_{timestamp}.yaml"), "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    with open(os.path.join(out_dir, f"hparams_{timestamp}.json"), "w") as f:
        json.dump(hparams, f, indent=4)


def build_model(config):
    """Instantiate BioVidOnlineClassifier from config dict."""
    model_cfg = config.get("model_params", {})
    lora_cfg = config.get("lora_params", {})
    optim_cfg = config.get("optimizer_params", {})
    data_cfg = config.get("data", {})

    return BioVidOnlineClassifier(
        num_classes=data_cfg.get("num_classes", 5),
        wan_checkpoint_dir=model_cfg["wan_checkpoint_dir"],
        vae_checkpoint=model_cfg["vae_checkpoint"],
        lora_rank=lora_cfg.get("rank", 4),
        lora_alpha=lora_cfg.get("alpha", 1.0),
        lora_target_modules=lora_cfg.get("target_modules", None),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", False),
        vae_in_channels=model_cfg.get("vae_in_channels", 16),
        xdit_in_channels=model_cfg.get("xdit_in_channels", 5120),
        feature_dim=model_cfg.get("feature_dim", 128),
        fusion_dim=model_cfg.get("fusion_dim", 128),
        shared_mlp_hidden_dims=model_cfg.get("shared_mlp_hidden_dims", None),
        shared_mlp_dropout=model_cfg.get("shared_mlp_dropout", 0.5),
        fusion_dropout=model_cfg.get("fusion_dropout", 0.0),
        fusion_activation=model_cfg.get("fusion_activation", "relu"),
        fusion_use_residual=model_cfg.get("fusion_use_residual", False),
        fusion_use_layernorm=model_cfg.get("fusion_use_layernorm", False),
        temporal_encoder_type=model_cfg.get("temporal_encoder_type", "gru"),
        temporal_encoder_nhead=model_cfg.get("temporal_encoder_nhead", 8),
        temporal_encoder_num_layers=model_cfg.get("temporal_encoder_num_layers", 2),
        temporal_encoder_dropout=model_cfg.get("temporal_encoder_dropout", 0.2),
        temporal_encoder_max_len=model_cfg.get("temporal_encoder_max_len", 512),
        temporal_encoder_use_layernorm=model_cfg.get("temporal_encoder_use_layernorm", False),
        temporal_pooling_type=model_cfg.get("temporal_pooling_type", "mean"),
        spatial_pool=model_cfg.get("spatial_pool", "mean"),
        lr_backbone=optim_cfg.get("lr_backbone", 5e-5),
        lr_head=optim_cfg.get("lr_head", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.01),
        label_smoothing=model_cfg.get("label_smoothing", 0.0),
        use_distance_penalty=model_cfg.get("use_distance_penalty", False),
        focal_gamma=model_cfg.get("focal_gamma", None),
        dit_feature_layer=model_cfg.get("dit_feature_layer", -1),
        dit_feature_layers=model_cfg.get("dit_feature_layers", None),
        dit_timestep=model_cfg.get("dit_timestep", 0.0),
        prompt_embeddings_path=model_cfg.get("prompt_embeddings_path", None),
        coral_alpha=model_cfg.get("coral_alpha", 1.0),
        ce_alpha=model_cfg.get("ce_alpha", 0.0),
        eval_head=model_cfg.get("eval_head", "coral"),
        ce_focal_gamma=model_cfg.get("ce_focal_gamma", None),
        ce_label_smoothing=model_cfg.get("ce_label_smoothing", 0.0),
        mixup_alpha=model_cfg.get("mixup_alpha", 0.0),
    )


def build_datamodule(config):
    """Instantiate BioVidOnlineDataModule from config dict."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    aug_cfg = config.get("augmentation", None)

    return BioVidOnlineDataModule(
        labels_csv=data_cfg["labels_csv"],
        frames_root=data_cfg["frames_root"],
        num_classes=data_cfg.get("num_classes", 5),
        batch_size=train_cfg.get("batch_size", 1),
        num_workers=train_cfg.get("num_workers", 4),
        resize=data_cfg.get("resize", 128),
        max_frames=data_cfg.get("max_frames", 129),
        sample_rate=data_cfg.get("sample_rate", 1),
        augmentation=aug_cfg,
        class_subset=data_cfg.get("class_subset", None),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Online LoRA Fine-tuning for Wan DiT — BioVid Pain Classification"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test"],
        help="train: training run, test: evaluate checkpoint",
    )
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint for test mode")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Validate required config fields
    model_params = config.get("model_params", {})
    data_cfg = config.get("data", {})
    if "wan_checkpoint_dir" not in model_params:
        raise ValueError("model_params.wan_checkpoint_dir is required")
    if "vae_checkpoint" not in model_params:
        raise ValueError("model_params.vae_checkpoint is required")
    if "labels_csv" not in data_cfg:
        raise ValueError("data.labels_csv is required")
    if "frames_root" not in data_cfg:
        raise ValueError("data.frames_root is required")

    train_cfg = config.get("train", {})
    run_name = config.get("run_name", "wan_lora_biovid")
    seed = train_cfg.get("seed", args.seed)
    monitor_metric = train_cfg.get("monitor_metric", "val_pain_QWK")
    max_epochs = train_cfg.get("max_epochs", 200)
    early_stop_patience = train_cfg.get("early_stop_patience", 30)
    precision = train_cfg.get("precision", "bf16-mixed")

    torch.manual_seed(seed)
    pl.seed_everything(seed)

    base, logs_dir, ckpt_dir, results_dir, timestamp = get_output_dir(
        run_name, args.output_root
    )
    save_hparams_to_dir(config, base, timestamp)

    model = build_model(config)
    dm = build_datamodule(config)
    dm.setup()

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        filename=f"{run_name}_best-{{epoch}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode="max" if "QWK" in monitor_metric else "min",
        save_top_k=1,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor_metric,
        mode="max" if "QWK" in monitor_metric else "min",
        patience=early_stop_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_summary = ModelSummary(max_depth=train_cfg.get("model_summary_depth", 3))
    callbacks = [ckpt_callback, early_stop, lr_monitor, model_summary]

    # Loggers
    tb_logger = TensorBoardLogger(logs_dir, name=run_name)
    csv_logger = CSVLogger(logs_dir, name=f"{run_name}_csv", version=timestamp)
    loggers = [tb_logger, csv_logger]

    accumulate = train_cfg.get("accumulate_grad_batches", 4)
    strategy = train_cfg.get("strategy", "auto")
    if args.n_gpus > 1 and strategy == "auto":
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = Trainer(
        log_every_n_steps=10,
        devices=args.n_gpus,
        accelerator="cpu" if args.n_gpus == 0 else "gpu",
        strategy=strategy,
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=accumulate,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=logs_dir,
        benchmark=args.n_gpus > 0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic="warn",
    )

    if args.mode == "train":
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        print(f"Best checkpoint: {ckpt_callback.best_model_path}")

        # Run test with best checkpoint
        test_loader = dm.test_dataloader()
        if test_loader is not None:
            test_results = trainer.test(
                model=None,
                dataloaders=test_loader,
                ckpt_path=ckpt_callback.best_model_path,
            )
            results_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to {results_file}")

        # Save LoRA weights separately
        lora_path = os.path.join(ckpt_dir, f"lora_weights_{timestamp}.pt")
        model.save_lora_only(lora_path)
        print(f"LoRA weights saved to {lora_path}")

    elif args.mode == "test":
        if not args.ckpt_path:
            raise ValueError("--ckpt_path required for test mode")
        test_loader = dm.test_dataloader() or dm.val_dataloader()
        if test_loader is not None:
            test_results = trainer.test(
                model=None, dataloaders=test_loader, ckpt_path=args.ckpt_path
            )
            results_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to {results_file}")

    print(f"\nAll outputs saved under: {base}")


if __name__ == "__main__":
    main()
