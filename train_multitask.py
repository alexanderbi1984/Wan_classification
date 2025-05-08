import argparse
import os
from datetime import datetime
import json
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.callbacks.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from data.syracuse import SyracuseDataModule
from data.biovid import BioVidDataModule
from classifier.multi_task_coral import MultiTaskCoralClassifier
from torch.utils.data import ConcatDataset

# Helper for timestamped output

def get_output_dir(run_name, base_dir="results", fold_idx=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(base_dir, f"{run_name}_{timestamp}")
    
    if fold_idx is not None:
        # Create fold-specific directories if in CV mode
        fold_dir = os.path.join(base, f"fold_{fold_idx}")
        logs = os.path.join(fold_dir, "logs"); os.makedirs(logs, exist_ok=True)
        ckpt = os.path.join(fold_dir, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
        results = os.path.join(fold_dir, "results"); os.makedirs(results, exist_ok=True)
    else:
        # Standard directories for regular training/testing
        logs = os.path.join(base, "logs"); os.makedirs(logs, exist_ok=True)
        ckpt = os.path.join(base, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
        results = os.path.join(base, "results"); os.makedirs(results, exist_ok=True)
    
    return base, logs, ckpt, results, timestamp

def save_hparams_to_dir(hparams, out_dir, timestamp):
    with open(os.path.join(out_dir, f"hparams_{timestamp}.yaml"), "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    with open(os.path.join(out_dir, f"hparams_{timestamp}.json"), "w") as f:
        json.dump(hparams, f, indent=4)

def setup_model_and_trainer(config, args, logs_dir, ckpt_dir, run_name, timestamp, input_dim):
    # --- Model Setup ---
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    optimizer_cfg = config.get("optimizer", {})
    
    # Add learning rate scheduler config if enabled
    lr_scheduler_cfg = config.get("lr_scheduler", {})
    use_lr_scheduler = lr_scheduler_cfg.get("use_scheduler", False)
    
    if use_lr_scheduler:
        lr_scheduler_params = {
            "use_lr_scheduler": True,
            "lr_factor": lr_scheduler_cfg.get("factor", 0.5),
            "lr_patience": lr_scheduler_cfg.get("patience", 5),
            "min_lr": lr_scheduler_cfg.get("min_lr", 1e-6),
            "monitor_metric": args.monitor_metric
        }
    else:
        lr_scheduler_params = {}
    
    hparams = {
        **model_cfg,
        **loss_cfg,
        **optimizer_cfg,
        **lr_scheduler_params,
        "input_dim": input_dim
    }
    model = MultiTaskCoralClassifier(**hparams)
    
    # --- Callbacks/Monitoring Setup ---
    monitor_metric = args.monitor_metric
    
    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(logs_dir, name=run_name)
    
    # Setup CSV logger for easy plotting
    csv_logger = CSVLogger(logs_dir, name=f"{run_name}_csv", version=timestamp)
    
    # Checkpoint callback
    checkpoint_name = f"{run_name}_best_{timestamp}" 
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        filename=checkpoint_name + "-{epoch}-{" + monitor_metric + ":.3f}",
        monitor=monitor_metric,
        mode="max" if "QWK" in monitor_metric else "min",
        save_top_k=1,
        verbose=True
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="max" if "QWK" in monitor_metric else "min",
        patience=config.get("early_stop_patience", 100),
        verbose=True
    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Model summary for architecture visualization
    model_summary = ModelSummary(max_depth=config.get("model_summary_depth", 2))
    
    # Set up callbacks
    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        lr_monitor,
        model_summary
    ]
    
    trainer = Trainer(
        log_every_n_steps=10,
        devices=args.n_gpus,
        accelerator="cpu" if args.n_gpus == 0 else "gpu",
        max_epochs=config.get("max_epochs", 75),
        precision=args.precision,
        logger=[tb_logger, csv_logger],  # Use both loggers
        callbacks=callbacks,
        default_root_dir=logs_dir,
        benchmark=True if args.n_gpus > 0 else False,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True,
        progress_bar_refresh_rate=config.get("progress_bar_refresh_rate", 1)
    )
    
    return model, trainer, ckpt_callback

def main():
    parser = argparse.ArgumentParser("Multi-Task Training & Evaluation (Joint Pain/Stimulus, Syracuse-centric)")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "cv"], help="Mode: train/test/cv")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for outputs/results")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--precision", type=str, default="32", help="Precision (e.g., 32, 16, bf16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to load model checkpoint in test mode")
    parser.add_argument("--monitor_metric", type=str, default="val_pain_QWK", 
                        choices=["val_pain_QWK", "val_pain_MAE"],
                        help="Which Syracuse metric to checkpoint/early-stop on: QWK or MAE")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config.get("run_name", "multitask_joint")
    
    # Set up the base directory first (common across all folds in CV mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, f"{run_name}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    save_hparams_to_dir(config, base_dir, timestamp)

    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    if args.mode == "cv":
        # Cross-validation mode: loop through 3 folds
        print("Running 3-fold cross-validation...")
        all_fold_results = []
        for fold_idx in range(3):
            print(f"\n=== STARTING FOLD {fold_idx} ===")
            # Create fold-specific directories
            _, logs_dir, ckpt_dir, results_dir, fold_timestamp = get_output_dir(run_name, base_dir, fold_idx)
            
            # Setup data modules with specific CV fold
            syracuse_dm = SyracuseDataModule(
                meta_path=config["syracuse_meta_path"],
                config_path=config.get("syracuse_config_path", None),
                batch_size=config.get("batch_size", 32),
                num_workers=args.num_workers,
                cv_fold=fold_idx,  # Use the current fold
                seed=args.seed,
                balanced_sampling=config.get("balanced_sampling", False),
                temporal_pooling=config.get("temporal_pooling", "mean"),
                flatten=config.get("flatten", True)
            )
            biovid_dm = BioVidDataModule(
                features_path=config["biovid_features_path"],
                meta_path=config["biovid_meta_path"],
                batch_size=config.get("batch_size", 32),
                num_workers=args.num_workers,
                split_ratio=config.get("split_ratio", 0.8),
                seed=args.seed,
                temporal_pooling=config.get("temporal_pooling", "mean"),
                flatten=config.get("flatten", True)
            )
            
            # Setup datasets and dataloaders
            syracuse_dm.setup(); biovid_dm.setup()
            train_ds = ConcatDataset([syracuse_dm.train_dataset, biovid_dm.train_dataset])
            val_ds = syracuse_dm.val_dataset
            
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=config.get("batch_size", 32),
                shuffle=True,
                num_workers=args.num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                num_workers=args.num_workers
            )
            
            # Get input shape for model initialization
            input_shape = syracuse_dm.example_shape
            flatten = config.get("flatten", True)
            temporal_pooling = config.get("temporal_pooling", "mean")
            if flatten and temporal_pooling != 'none':
                if not (isinstance(input_shape, tuple) and len(input_shape) == 1):
                    raise ValueError(f"[Config Error] flatten=True, temporal_pooling={temporal_pooling}, but dataset shape is {input_shape} (expected 1D). Check setup.")
            if isinstance(input_shape, tuple):
                input_dim = 1
                for x in input_shape:
                    input_dim *= x
            else:
                input_dim = int(input_shape)
                
            # Setup model and trainer for this fold
            model, trainer, ckpt_callback = setup_model_and_trainer(
                config, args, logs_dir, ckpt_dir, f"{run_name}_fold{fold_idx}", fold_timestamp, input_dim
            )
            
            # Train model for this fold
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Test model for this fold using the best checkpoint
            print(f"Fold {fold_idx}: Running test step with best checkpoint")
            fold_results = trainer.test(model=None, dataloaders=val_loader, ckpt_path=ckpt_callback.best_model_path)
            
            # Save fold results
            fold_results_filename = os.path.join(results_dir, f"fold_{fold_idx}_results.json")
            with open(fold_results_filename, "w") as f:
                json.dump(fold_results, f, indent=4)
            
            # Collect results for aggregation
            if fold_results and isinstance(fold_results, list) and len(fold_results) > 0:
                all_fold_results.append(fold_results[0])
                print(f"Fold {fold_idx} completed, results saved to {fold_results_filename}")
        
        # Aggregate and save overall CV results
        if all_fold_results:
            # Get all metrics from the first fold result as reference
            all_metrics = list(all_fold_results[0].keys())
            
            # Prepare aggregated results
            aggregated_results = {
                "num_completed_folds": len(all_fold_results),
                "individual_fold_results": all_fold_results,
            }
            
            # Calculate mean and std for each metric
            for metric in all_metrics:
                values = [fold[metric] for fold in all_fold_results]
                aggregated_results[f"{metric}_mean"] = float(np.mean(values))
                aggregated_results[f"{metric}_std"] = float(np.std(values))
            
            # Save the aggregated results
            cv_results_filename = os.path.join(base_dir, f"cv_aggregated_results.json")
            with open(cv_results_filename, "w") as f:
                json.dump(aggregated_results, f, indent=4)
            
            # Print summary of key metrics
            print("\n===== 3-FOLD CROSS-VALIDATION SUMMARY =====")
            for metric in all_metrics:
                if any(kw in metric for kw in ["QWK", "MAE", "Accuracy", "acc"]):
                    print(f"  {metric}: {aggregated_results[f'{metric}_mean']:.4f} ± {aggregated_results[f'{metric}_std']:.4f}")
            print("============================================")
            print(f"Full CV results saved to {cv_results_filename}")
        print(f"All CV logs and results saved under {base_dir}")
        
    else:
        # Regular train or test mode (existing functionality)
        base_dir, logs_dir, ckpt_dir, results_dir, timestamp = get_output_dir(run_name, args.output_root)
        save_hparams_to_dir(config, base_dir, timestamp)
        
        # --- Multitask DataModule Setup ---
        # Always construct joint/fused training: combine (Syracuse) and (BioVid)
        print("Setting up joint (multitask) data loader for Syracuse (pain, main) & BioVid (stimulus)...")

        syracuse_dm = SyracuseDataModule(
            meta_path=config["syracuse_meta_path"],
            config_path=config.get("syracuse_config_path", None),
            batch_size=config.get("batch_size", 32),
            num_workers=args.num_workers,
            cv_fold=config.get("cv_fold", 0),
            seed=args.seed,
            balanced_sampling=config.get("balanced_sampling", False),
            temporal_pooling=config.get("temporal_pooling", "mean"),
            flatten=config.get("flatten", True)
        )
        biovid_dm = BioVidDataModule(
            features_path=config["biovid_features_path"],
            meta_path=config["biovid_meta_path"],
            batch_size=config.get("batch_size", 32),
            num_workers=args.num_workers,
            split_ratio=config.get("split_ratio", 0.8),
            seed=args.seed,
            temporal_pooling=config.get("temporal_pooling", "mean"),
            flatten=config.get("flatten", True)
        )
        syracuse_dm.setup(); biovid_dm.setup()
        # Concatenate train datasets from both modules
        train_ds = ConcatDataset([syracuse_dm.train_dataset, biovid_dm.train_dataset])
        val_ds = syracuse_dm.val_dataset  # Validation always evaluated on Syracuse (pain) only
        # Use the batch size and dataloader params from config/args
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=args.num_workers
        )
        # --- Flatten shape assertion ---
        input_shape = syracuse_dm.example_shape
        flatten = config.get("flatten", True)
        temporal_pooling = config.get("temporal_pooling", "mean")
        if flatten and temporal_pooling != 'none':
            if not (isinstance(input_shape, tuple) and len(input_shape) == 1):
                raise ValueError(f"[Config Error] flatten=True, temporal_pooling={temporal_pooling}, but dataset shape is {input_shape} (expected 1D). Check setup.")
        if isinstance(input_shape, tuple):
            input_dim = 1
            for x in input_shape:
                input_dim *= x
        else:
            input_dim = int(input_shape)

        # Set up model and trainer
        model, trainer, ckpt_callback = setup_model_and_trainer(
            config, args, logs_dir, ckpt_dir, run_name, timestamp, input_dim
        )
        save_hparams_to_dir(model.hparams, base_dir, timestamp + "_final")

        # --- Training ---
        if args.mode == "train":
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            print(f"Best checkpoint at: {ckpt_callback.best_model_path}")
        # --- Testing (pain task only; Syracuse validation/test set): ---
        if args.mode in ["test", "train"]:
            ckpt_path = args.ckpt_path
            if not ckpt_path:
                ckpt_path = ckpt_callback.best_model_path or None
            print("Running test step with the best checkpoint... (metrics: Syracuse pain)")
            test_results = trainer.test(model if not ckpt_path else None, dataloaders=val_loader, ckpt_path=ckpt_path)
            results_filename = os.path.join(results_dir, f"test_results_{timestamp}.json")
            with open(results_filename, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to {results_filename}")
            # Print topline metrics summary
            if test_results and isinstance(test_results, list) and len(test_results)>0:
                metric_row = test_results[0]
                toprint = [k for k in metric_row if any(kw in k for kw in ["QWK", "MAE", "Accuracy", "acc"]) ]
                print("\n===== SYRACUSE PAIN (MAIN) EVALUATION SUMMARY =====")
                for k in toprint:
                    print(f"  {k}: {metric_row[k]}")
                print("===============================================")
        print(f"All logs, checkpoints, and configs are saved under {base_dir}")

if __name__ == "__main__":
    main() 