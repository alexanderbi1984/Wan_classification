import argparse
import os
from datetime import datetime
import json
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from data.biovid import BioVidDataModule
from data.combined_task_wrapper import CombinedTaskDatasetWrapper
from classifier.biovid_coral import BioVidCoralClassifier
from torch.utils.data import DataLoader

def get_output_dir(run_name, base_dir="results"):
    """Create output directories for logs, checkpoints, and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(base_dir, f"{run_name}_{timestamp}")
    logs = os.path.join(base, "logs")
    ckpt = os.path.join(base, "checkpoints")
    results = os.path.join(base, "results")
    
    for d in [logs, ckpt, results]:
        os.makedirs(d, exist_ok=True)
    
    return base, logs, ckpt, results, timestamp

def save_hparams_to_dir(hparams, out_dir, timestamp):
    """Save hyperparameters in both YAML and JSON formats."""
    with open(os.path.join(out_dir, f"hparams_{timestamp}.yaml"), "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    with open(os.path.join(out_dir, f"hparams_{timestamp}.json"), "w") as f:
        json.dump(hparams, f, indent=4)

def detect_input_shape(dm, flatten=True, temporal_pooling='mean'):
    """Detect input shape from the data module."""
    input_shape = None
    
    # Try to get shape from data module
    if hasattr(dm, 'example_shape') and dm.example_shape is not None:
        input_shape = dm.example_shape
    elif hasattr(dm, 'train_dataset') and dm.train_dataset and len(dm.train_dataset) > 0:
        if hasattr(dm.train_dataset, 'example_shape') and dm.train_dataset.example_shape is not None:
            input_shape = dm.train_dataset.example_shape
    
    if input_shape is None:
        raise ValueError(f"Could not detect input_shape from data module")
    
    # Calculate input dimension
    if isinstance(input_shape, tuple):
        input_dim = 1
        for x in input_shape:
            input_dim *= x
    else:
        input_dim = int(input_shape)
    
    print(f"[INFO] Determined input_dim for model: {input_dim} (from shape: {input_shape})")
    return input_shape, input_dim

def setup_model_and_trainer(config, args, logs_dir, ckpt_dir, run_name, timestamp, input_dim):
    """Set up the model and trainer with callbacks."""
    model_cfg = config.get("model_params", {})
    optimizer_cfg = config.get("optimizer_params", {})
    lr_scheduler_cfg = config.get("lr_scheduler_params", {})
    trainer_cfg = config.get("trainer_params", {})
    biovid_cfg = config.get("biovid_settings", {})

    # Prepare model hyperparameters
    hparams_model = {**model_cfg}
    
    # Set up for single-task (BioVid only)
    # Set pain classes to minimum required (2) since we'll use -1 labels to disable the task
    hparams_model["num_pain_classes"] = 2  # Minimum required, task disabled via -1 labels
    hparams_model["num_stimulus_classes"] = biovid_cfg.get("num_classes_stimulus", 5)
    
    # Add optimizer settings
    if 'optimizer_type' in optimizer_cfg:
        optimizer_cfg['optimizer_name'] = optimizer_cfg.pop('optimizer_type')
    hparams_model.update(optimizer_cfg)

    # Add LR scheduler settings if enabled
    if lr_scheduler_cfg.get("use_scheduler", False):
        hparams_model.update({
            "use_lr_scheduler": True,
            "lr_factor": lr_scheduler_cfg.get("factor", 0.1),
            "lr_patience": lr_scheduler_cfg.get("patience", 10),
            "min_lr": lr_scheduler_cfg.get("min_lr", 1e-6)
        })
    else:
        hparams_model["use_lr_scheduler"] = False
    
    hparams_model["input_dim"] = input_dim
    
    # Initialize model using BioVidCoralClassifier
    model = BioVidCoralClassifier(**hparams_model)
    
    # Set up loggers
    tb_logger = TensorBoardLogger(logs_dir, name=run_name)
    csv_logger = CSVLogger(logs_dir, name=f"{run_name}_csv", version=timestamp)
    
    # Set up callbacks
    monitor_metric = "val_stim_QWK"  # Always use stimulus metrics for BioVid
    mode = "max"  # QWK is always maximized
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}_best_{timestamp}" + "-{epoch}-{" + monitor_metric + ":.3f}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=mode,
        patience=trainer_cfg.get("early_stop_patience", 15),
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model_summary = ModelSummary(max_depth=trainer_cfg.get("model_summary_depth", 2))
    
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        model_summary
    ]
    
    # Initialize trainer
    trainer = Trainer(
        log_every_n_steps=10,
        devices=args.n_gpus,
        accelerator="cpu" if args.n_gpus == 0 else "gpu",
        max_epochs=trainer_cfg.get("max_epochs", 100),
        precision=args.precision,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        default_root_dir=logs_dir,
        benchmark=True if args.n_gpus > 0 else False,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return model, trainer, checkpoint_callback

def main():
    parser = argparse.ArgumentParser("BioVid Pain Level Classification Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train/test")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for outputs")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--precision", type=str, default="32", help="Precision (32, 16, bf16)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint for testing")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config.get("run_name", "biovid_baseline")
    trainer_cfg = config.get("trainer_params", {})
    biovid_cfg = config.get("biovid_settings", {})
    batch_size = trainer_cfg.get("batch_size", 32)

    # Set up directories
    base_dir, logs_dir, ckpt_dir, results_dir, timestamp = get_output_dir(run_name, args.output_root)
    save_hparams_to_dir(config, base_dir, timestamp)

    # Set random seed
    seed = trainer_cfg.get("seed", 42)
    pl.seed_everything(seed)
    torch.manual_seed(seed)

    # Initialize data module
    biovid_dm = BioVidDataModule(
        features_path=biovid_cfg.get("features_path"),
        meta_path=biovid_cfg.get("meta_path"),
        batch_size=batch_size,
        num_workers=args.num_workers,
        split_ratio=biovid_cfg.get("split_ratio", 0.8),
        seed=seed,
        temporal_pooling=biovid_cfg.get("temporal_pooling", "mean"),
        flatten=biovid_cfg.get("flatten", True)
    )
    biovid_dm.setup()

    # Wrap datasets for multi-task compatibility
    wrapped_train_dataset = CombinedTaskDatasetWrapper(biovid_dm.train_dataset, task_name='stimulus')
    wrapped_val_dataset = CombinedTaskDatasetWrapper(biovid_dm.val_dataset, task_name='stimulus')

    # Create data loaders with wrapped datasets
    train_loader = DataLoader(
        wrapped_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        wrapped_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Get input dimensions from wrapped dataset
    input_shape, input_dim = detect_input_shape(
        wrapped_train_dataset,
        flatten=biovid_cfg.get("flatten", True),
        temporal_pooling=biovid_cfg.get("temporal_pooling", "mean")
    )

    # Set up model and trainer
    model, trainer, ckpt_callback = setup_model_and_trainer(
        config, args, logs_dir, ckpt_dir, run_name, timestamp, input_dim
    )

    # Training
    if args.mode == "train":
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"Best checkpoint saved at: {ckpt_callback.best_model_path}")

    # Testing
    if args.mode in ["test", "train"]:
        ckpt_path = args.ckpt_path if args.mode == "test" else ckpt_callback.best_model_path
        if not ckpt_path and args.mode == "test":
            raise ValueError("ckpt_path must be provided for test mode")

        print(f"Running test step with checkpoint: {ckpt_path}")
        test_results = trainer.test(
            model=None,
            dataloaders=val_loader,
            ckpt_path=ckpt_path
        )

        # Save test results
        results_filename = os.path.join(results_dir, f"test_results_{timestamp}.json")
        with open(results_filename, "w") as f:
            json.dump(test_results, f, indent=4)

        # Print key metrics
        if test_results and isinstance(test_results, list) and len(test_results) > 0:
            metrics = test_results[0]
            print("\n===== BIOVID EVALUATION SUMMARY =====")
            for k, v in metrics.items():
                if any(kw in k for kw in ["QWK", "MAE", "Accuracy"]):
                    print(f"  {k}: {v}")
            print("====================================")

        print(f"Test results saved to {results_filename}")
    
    print(f"All outputs saved under {base_dir}")

if __name__ == "__main__":
    main() 