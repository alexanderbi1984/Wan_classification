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
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from data.syracuse import SyracuseDataModule
from data.biovid import BioVidDataModule
from data.combined_task_wrapper import CombinedTaskDatasetWrapper
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

def detect_input_shape(primary_dm, fallback_dm=None, flatten=True, temporal_pooling='mean'):
    """
    Robustly detect input shape from data modules with clear error messages.
    
    Args:
        primary_dm: Primary data module to check first (e.g., syracuse_dm)
        fallback_dm: Secondary data module to check as fallback (e.g., biovid_dm)
        flatten: Whether datasets are expected to produce flattened outputs
        temporal_pooling: The temporal pooling strategy being used
        
    Returns:
        Tuple: (input_shape, input_dim) - both the raw shape and calculated dimension
        
    Raises:
        ValueError: If input shape cannot be determined or validation fails
    """
    input_shape = None
    
    # Try to get shape from primary or fallback data modules
    for dm in [primary_dm, fallback_dm]:
        if dm is None:
            continue
            
        # Try direct example_shape first
        if hasattr(dm, 'example_shape') and dm.example_shape is not None:
            input_shape = dm.example_shape
            print(f"[INFO] Found example_shape from {dm.__class__.__name__}: {input_shape}")
            break
            
        # Fall back to dataset example shape
        if hasattr(dm, 'train_dataset') and dm.train_dataset and len(dm.train_dataset) > 0:
            if hasattr(dm.train_dataset, 'example_shape') and dm.train_dataset.example_shape is not None:
                input_shape = dm.train_dataset.example_shape
                print(f"[INFO] Found example_shape from {dm.__class__.__name__}.train_dataset: {input_shape}")
                break
    
    # If we still don't have a shape, raise an error
    if input_shape is None:
        modules_str = f"{primary_dm.__class__.__name__}, {fallback_dm.__class__.__name__ if fallback_dm else 'None'}"
        raise ValueError(f"Could not detect input_shape from data modules: {modules_str}")
    
    # Validate shape based on flattening expectations
    if flatten and temporal_pooling != 'none':
        # For flattened data with pooling, we expect a 1D shape
        if isinstance(input_shape, tuple) and len(input_shape) > 1:
            print(f"[WARNING] Expected 1D shape with flatten={flatten} and temporal_pooling='{temporal_pooling}', "
                  f"but got {input_shape}. Will flatten automatically.")
    
    # Calculate input dimension
    if isinstance(input_shape, tuple):
        input_dim = 1
        for x in input_shape:
            input_dim *= x
    else:
        # Should be int if already flattened to a single dim
        input_dim = int(input_shape)
    
    print(f"[INFO] Determined input_dim for model: {input_dim} (from shape: {input_shape})")
    
    # Basic sanity check
    if input_dim <= 0:
        raise ValueError(f"Calculated input_dim is {input_dim}, which is invalid. Check dataset implementation.")
        
    return input_shape, input_dim

def setup_model_and_trainer(config, args, logs_dir, ckpt_dir, run_name, timestamp, input_dim):
    # --- Model Setup ---
    model_cfg = config.get("model_params", {})
    optimizer_cfg = config.get("optimizer_params", {})
    lr_scheduler_cfg = config.get("lr_scheduler_params", {})
    trainer_cfg = config.get("trainer_params", {})

    use_lr_scheduler = lr_scheduler_cfg.get("use_scheduler", False)
    
    hparams_model = {**model_cfg} # Start with model-specific hparams
    
    # Add num_classes from specific data settings to model hparams
    syracuse_cfg = config.get("syracuse_settings", {})
    biovid_cfg = config.get("biovid_settings", {})
    hparams_model["num_pain_classes"] = syracuse_cfg.get("num_classes_pain") 
    hparams_model["num_stimulus_classes"] = biovid_cfg.get("num_classes_stimulus")

    # Add optimizer specific hparams (like learning_rate, optimizer_type, weight_decay)
    if 'optimizer_type' in optimizer_cfg:
        optimizer_cfg['optimizer_name'] = optimizer_cfg.pop('optimizer_type')
    hparams_model.update(optimizer_cfg)

    if use_lr_scheduler:
        # Pass LR scheduler specific hparams (factor, patience, min_lr)
        # and also the metric to monitor for the scheduler
        hparams_model.update({
            "use_lr_scheduler": True,
            "lr_factor": lr_scheduler_cfg.get("factor", 0.1), # Adjusted default
            "lr_patience": lr_scheduler_cfg.get("patience", 10), # Adjusted default
            "min_lr": lr_scheduler_cfg.get("min_lr", 1e-6),
            "monitor_metric": trainer_cfg.get("monitor_metric", args.monitor_metric) # Use trainer_cfg or fallback to args
        })
    else:
        hparams_model["use_lr_scheduler"] = False
    
    hparams_model["input_dim"] = input_dim
    
    # Ensure all required hparams for MultiTaskCoralClassifier are present
    # Example: num_pain_classes, num_stimulus_classes, learning_rate, etc.
    # MultiTaskCoralClassifier will raise an error if essential ones are missing.
    model = MultiTaskCoralClassifier(**hparams_model)
    
    # --- Callbacks/Monitoring Setup ---
    # Use monitor_metric from trainer_config, fallback to args if not present
    monitor_metric = trainer_cfg.get("monitor_metric", args.monitor_metric)
    
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
        patience=trainer_cfg.get("early_stop_patience", 20), # From trainer_params
        verbose=True
    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Model summary for architecture visualization
    model_summary = ModelSummary(max_depth=trainer_cfg.get("model_summary_depth", 2)) # From trainer_params
    
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
        max_epochs=trainer_cfg.get("max_epochs", 75), # From trainer_params
        precision=args.precision,
        logger=[tb_logger, csv_logger],  # Use both loggers
        callbacks=callbacks,
        default_root_dir=logs_dir,
        benchmark=True if args.n_gpus > 0 else False,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return model, trainer, ckpt_callback

def main():
    parser = argparse.ArgumentParser("Multi-Task Training & Evaluation (Joint Pain/Stimulus, Syracuse-centric)")
    parser.add_argument("--config", type=str, required=True, help="Path to the unified experiment config YAML")
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

    run_name = config.get("run_name", "multitask_default_run")
    syracuse_cfg = config.get("syracuse_settings", {})
    biovid_cfg = config.get("biovid_settings", {})
    trainer_cfg = config.get("trainer_params", {})
    # batch_size is now under trainer_cfg
    batch_size = trainer_cfg.get("batch_size", 32) 
    
    # Set up the base directory first (common across all folds in CV mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, f"{run_name}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    save_hparams_to_dir(config, base_dir, timestamp) # Save the whole new config

    # Use seed from trainer_params in config, fallback to args.seed
    seed_to_use = trainer_cfg.get("seed", args.seed)
    torch.manual_seed(seed_to_use)
    pl.seed_everything(seed_to_use)

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
                meta_path=syracuse_cfg.get("meta_path"),
                task=syracuse_cfg.get("task", "classification"), # Pass task
                thresholds=syracuse_cfg.get("thresholds", None), # Pass thresholds
                batch_size=batch_size, # Use batch_size from trainer_cfg
                num_workers=args.num_workers,
                cv_fold=fold_idx,
                seed=seed_to_use, # Use consistent seed
                balanced_sampling=syracuse_cfg.get("balanced_sampling", False),
                temporal_pooling=syracuse_cfg.get("temporal_pooling", "mean"),
                flatten=syracuse_cfg.get("flatten", True)
            )
            biovid_dm = BioVidDataModule(
                features_path=biovid_cfg.get("features_path"),
                meta_path=biovid_cfg.get("meta_path"),
                batch_size=batch_size, # Use batch_size from trainer_cfg
                num_workers=args.num_workers,
                split_ratio=biovid_cfg.get("split_ratio", 0.8),
                seed=seed_to_use, # Use consistent seed
                temporal_pooling=biovid_cfg.get("temporal_pooling", "mean"),
                flatten=biovid_cfg.get("flatten", True)
            )
            
            # Setup original datasets first
            syracuse_dm.setup(); biovid_dm.setup()
            
            # --- Get input_dim from the original Syracuse dataset before wrapping ---
            # This ensures input_dim is based on the raw feature shape.
            input_shape, input_dim = detect_input_shape(
                primary_dm=syracuse_dm, 
                fallback_dm=biovid_dm,
                flatten=syracuse_cfg.get("flatten", True),
                temporal_pooling=syracuse_cfg.get("temporal_pooling", "mean")
            )

            # --- Wrap datasets for multi-task learning ---
            wrapped_syracuse_train = CombinedTaskDatasetWrapper(syracuse_dm.train_dataset, task_name='pain_level')
            wrapped_biovid_train = CombinedTaskDatasetWrapper(biovid_dm.train_dataset, task_name='stimulus')
            
            # Validation dataset (Syracuse only, but wrapped)
            # Ensure val_dataset exists. If cv_fold >= 3, syracuse_dm.val_dataset might be empty.
            if syracuse_dm.val_dataset and len(syracuse_dm.val_dataset) > 0:
                wrapped_syracuse_val = CombinedTaskDatasetWrapper(syracuse_dm.val_dataset, task_name='pain_level')
                val_ds_for_loader = wrapped_syracuse_val
            else:
                print("[WARN] Syracuse validation dataset is empty. Using a dummy empty list for val_loader. This is expected if cv_fold >= 3 or val split is empty.")
                val_ds_for_loader = [] # Trainer handles empty val_loader gracefully

            # Create ConcatDataset for training using wrapped datasets
            train_ds = ConcatDataset([wrapped_syracuse_train, wrapped_biovid_train])
            val_ds = val_ds_for_loader # Use the wrapped (or empty) val_ds
            
            # --- Dataloaders ---
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size, # Use batch_size from trainer_cfg
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=None # Use default collate, wrapper ensures consistent items
            )
            # Only create val_loader if val_ds is not empty to avoid DataLoader errors with empty datasets.
            if val_ds and len(val_ds) > 0:
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=batch_size, # Use batch_size from trainer_cfg
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=None
                )
            else:
                val_loader = None # Pass None to trainer.fit if no validation data
                            
            # Setup model and trainer for this fold
            model, trainer, ckpt_callback = setup_model_and_trainer(
                config, args, logs_dir, ckpt_dir, f"{run_name}_fold{fold_idx}", fold_timestamp, input_dim
            )
            
            # Train model for this fold
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Test model for this fold using the best checkpoint (if val_loader exists)
            if val_loader:
                print(f"Fold {fold_idx}: Running test step with best checkpoint on validation data")
                # Initial test call to get standard metrics
                fold_results_list = trainer.test(model=None, dataloaders=val_loader, ckpt_path=ckpt_callback.best_model_path)
                
                current_fold_metrics = {}
                if fold_results_list and isinstance(fold_results_list, list) and len(fold_results_list) > 0:
                    current_fold_metrics = fold_results_list[0].copy() # Work on a copy

                    # --- Confusion Matrix Calculation ---
                    best_ckpt_path = ckpt_callback.best_model_path
                    if best_ckpt_path and os.path.exists(best_ckpt_path):
                        print(f"Fold {fold_idx}: Reloading best model from {best_ckpt_path} for Confusion Matrix calculation.")
                        
                        # Re-create hparams for loading the model
                        model_cfg_fold = config.get("model_params", {})
                        optimizer_cfg_fold = config.get("optimizer_params", {})
                        lr_scheduler_cfg_fold = config.get("lr_scheduler_params", {})
                        # trainer_cfg_fold for monitor_metric, already available as trainer_cfg from outer scope
                        
                        hparams_for_load = {**model_cfg_fold}
                        # syracuse_cfg and biovid_cfg are from the outer scope, directly usable
                        hparams_for_load["num_pain_classes"] = syracuse_cfg.get("num_classes_pain")
                        hparams_for_load["num_stimulus_classes"] = biovid_cfg.get("num_classes_stimulus")
                        
                        # Correctly handle optimizer_name
                        temp_optimizer_cfg = optimizer_cfg_fold.copy()
                        if 'optimizer_type' in temp_optimizer_cfg:
                            temp_optimizer_cfg['optimizer_name'] = temp_optimizer_cfg.pop('optimizer_type')
                        hparams_for_load.update(temp_optimizer_cfg)

                        if lr_scheduler_cfg_fold.get("use_lr_scheduler", False): # Use lr_scheduler_cfg_fold here
                            hparams_for_load.update({
                                "use_lr_scheduler": True,
                                "lr_factor": lr_scheduler_cfg_fold.get("factor", 0.1), # Use lr_scheduler_cfg_fold
                                "lr_patience": lr_scheduler_cfg_fold.get("patience", 10), # Use lr_scheduler_cfg_fold
                                "min_lr": lr_scheduler_cfg_fold.get("min_lr", 1e-6), # Use lr_scheduler_cfg_fold
                                "monitor_metric": trainer_cfg.get("monitor_metric", args.monitor_metric)
                            })
                        else:
                            hparams_for_load["use_lr_scheduler"] = False
                        
                        hparams_for_load["input_dim"] = input_dim # 'input_dim' is from the current fold's scope

                        try:
                            loaded_model = MultiTaskCoralClassifier.load_from_checkpoint(
                                checkpoint_path=best_ckpt_path,
                                map_location=torch.device('cuda' if args.n_gpus > 0 else 'cpu'),
                                **hparams_for_load
                            )
                            loaded_model.eval()
                            loaded_model.freeze()

                            cm_trainer = Trainer(
                                devices=args.n_gpus,
                                accelerator="cpu" if args.n_gpus == 0 else "gpu",
                                logger=False,
                                callbacks=[],
                                enable_progress_bar=False,
                                enable_model_summary=False,
                                num_sanity_val_steps=0 # Avoid issues with CM metrics during sanity check
                            )

                            print(f"Fold {fold_idx}: Running dedicated test pass for confusion matrices...")
                            cm_trainer.test(model=loaded_model, dataloaders=val_loader, verbose=False)

                            if hasattr(loaded_model, 'test_pain_cm') and syracuse_cfg.get("num_classes_pain", 0) > 0:
                                pain_cm_tensor = loaded_model.test_pain_cm.compute().cpu().tolist()
                                current_fold_metrics['test_fold_pain_ConfusionMatrix'] = pain_cm_tensor
                                loaded_model.test_pain_cm.reset()
                                print(f"Fold {fold_idx}: Pain Confusion Matrix calculated: {pain_cm_tensor}")

                            if hasattr(loaded_model, 'test_stim_cm') and biovid_cfg.get("num_classes_stimulus", 0) > 0:
                                stim_cm_tensor = loaded_model.test_stim_cm.compute().cpu().tolist()
                                current_fold_metrics['test_fold_stim_ConfusionMatrix'] = stim_cm_tensor
                                loaded_model.test_stim_cm.reset()
                                print(f"Fold {fold_idx}: Stimulus Confusion Matrix calculated: {stim_cm_tensor}")
                        
                        except Exception as e:
                            print(f"Fold {fold_idx}: Error during CM calculation: {e}. Skipping CM for this fold.")
                            current_fold_metrics['test_fold_pain_ConfusionMatrix'] = f"Error: {e}"
                            current_fold_metrics['test_fold_stim_ConfusionMatrix'] = f"Error: {e}"

                    else:
                        print(f"Fold {fold_idx}: Best checkpoint path not found or invalid ({best_ckpt_path}). Skipping CM calculation.")
                        current_fold_metrics['test_fold_pain_ConfusionMatrix'] = "Skipped (no best_ckpt_path)"
                        current_fold_metrics['test_fold_stim_ConfusionMatrix'] = "Skipped (no best_ckpt_path)"
                
                # Save fold results (now potentially including CMs)
                fold_results_filename = os.path.join(results_dir, f"fold_{fold_idx}_results.json")
                with open(fold_results_filename, "w") as f:
                    # Save the (potentially modified) current_fold_metrics
                    json.dump(current_fold_metrics if current_fold_metrics else fold_results_list, f, indent=4)
                
                if current_fold_metrics: # Check if current_fold_metrics was populated
                    all_fold_results.append(current_fold_metrics) 
                    print(f"Fold {fold_idx} completed, results saved to {fold_results_filename}")
                elif fold_results_list: # Fallback if current_fold_metrics stayed empty (e.g. test returned empty)
                    all_fold_results.append(fold_results_list[0])
                    print(f"Fold {fold_idx} completed (CMs might be missing), results saved to {fold_results_filename}")

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
        # Regular train or test mode
        # logs_dir, ckpt_dir, results_dir were already being created inside get_output_dir with base_dir
        # base_dir is already created before the if/else for args.mode
        _, logs_dir, ckpt_dir, results_dir, timestamp = get_output_dir(run_name, base_dir) # timestamp from here or outer scope? use outer.
        # Re-fetch timestamp if not CV mode for consistency, though base_dir uses an earlier one.
        # For simplicity, we can stick to the initial timestamp for the overall run for hparams saving.
        # save_hparams_to_dir(config, base_dir, timestamp) # Already saved before mode check
        
        # --- Multitask DataModule Setup ---
        print("Setting up joint (multitask) data loader for Syracuse (pain, main) & BioVid (stimulus)...")

        syracuse_dm = SyracuseDataModule(
            meta_path=syracuse_cfg.get("meta_path"),
            task=syracuse_cfg.get("task", "classification"), # Pass task
            thresholds=syracuse_cfg.get("thresholds", None), # Pass thresholds
            batch_size=batch_size, # Use batch_size from trainer_cfg
            num_workers=args.num_workers,
            cv_fold=syracuse_cfg.get("cv_fold", 0), # For non-CV mode, allow config to specify default split
            seed=seed_to_use, # Use consistent seed
            balanced_sampling=syracuse_cfg.get("balanced_sampling", False),
            temporal_pooling=syracuse_cfg.get("temporal_pooling", "mean"),
            flatten=syracuse_cfg.get("flatten", True)
        )
        biovid_dm = BioVidDataModule(
            features_path=biovid_cfg.get("features_path"),
            meta_path=biovid_cfg.get("meta_path"),
            batch_size=batch_size, # Use batch_size from trainer_cfg
            num_workers=args.num_workers,
            split_ratio=biovid_cfg.get("split_ratio", 0.8),
            seed=seed_to_use, # Use consistent seed
            temporal_pooling=biovid_cfg.get("temporal_pooling", "mean"),
            flatten=biovid_cfg.get("flatten", True)
        )
        syracuse_dm.setup(); biovid_dm.setup()

        # --- Get input_dim from the original Syracuse dataset before wrapping ---
        input_shape, input_dim = detect_input_shape(
            primary_dm=syracuse_dm, 
            fallback_dm=biovid_dm,
            flatten=syracuse_cfg.get("flatten", True),
            temporal_pooling=syracuse_cfg.get("temporal_pooling", "mean")
        )

        # --- Wrap datasets ---
        wrapped_syracuse_train = CombinedTaskDatasetWrapper(syracuse_dm.train_dataset, task_name='pain_level')
        wrapped_biovid_train = CombinedTaskDatasetWrapper(biovid_dm.train_dataset, task_name='stimulus')
        
        if syracuse_dm.val_dataset and len(syracuse_dm.val_dataset) > 0:
            wrapped_syracuse_val = CombinedTaskDatasetWrapper(syracuse_dm.val_dataset, task_name='pain_level')
            val_ds_for_loader = wrapped_syracuse_val
        else:
            print("[WARN] Syracuse validation dataset is empty for train/test mode. Using dummy empty list for val_loader.")
            val_ds_for_loader = []

        train_ds = ConcatDataset([wrapped_syracuse_train, wrapped_biovid_train])
        val_ds = val_ds_for_loader
        
        # --- Dataloaders ---
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size, # Use batch_size from trainer_cfg
            shuffle=True,
            num_workers=args.num_workers
        )
        if val_ds and len(val_ds) > 0:
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=batch_size, # Use batch_size from trainer_cfg
                shuffle=False,
                num_workers=args.num_workers
            )
        else:
            val_loader = None

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
            if val_loader: # Only test if there is a validation set
                ckpt_path = args.ckpt_path
                if not ckpt_path and args.mode == "train": # If training, use best from this run
                    ckpt_path = ckpt_callback.best_model_path
                elif not ckpt_path and args.mode == "test": # If testing and no path, error or use last?
                     raise ValueError("ckpt_path must be provided for mode=test if not chained after training.")

                print(f"Running test step with checkpoint: {ckpt_path} (metrics: Syracuse pain)")
                test_results = trainer.test(model if args.mode == "train" and not ckpt_path else None, dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
                results_filename = os.path.join(results_dir, f"test_results_{timestamp}.json")
                with open(results_filename, "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Test results saved to {results_filename}")
                if test_results and isinstance(test_results, list) and len(test_results)>0:
                    metric_row = test_results[0]
                    toprint = [k for k in metric_row if any(kw in k for kw in ["QWK", "MAE", "Accuracy", "acc"]) ]
                    print("\n===== SYRACUSE PAIN (MAIN) EVALUATION SUMMARY =====")
                    for k in toprint:
                        print(f"  {k}: {metric_row[k]}")
                    print("===============================================")
            else:
                print("Skipping test step as validation dataloader is empty.")
        print(f"All logs, checkpoints, and configs are saved under {base_dir}")

if __name__ == "__main__":
    main() 