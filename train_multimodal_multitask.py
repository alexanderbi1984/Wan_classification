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
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging, GradientAccumulationScheduler
try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None
try:
    from pytorch_lightning.loggers import CometLogger
except ImportError:
    CometLogger = None
from data.syracuse import SyracuseDataModule
from data.biovid import BioVidDataModule
from data.combined_task_wrapper import CombinedTaskDatasetWrapper
from data.multimodal_datasets import SyracuseMultimodalDataModule, BioVidMultimodalDataset
from classifier.multimodal import MultimodalMultiTaskCoralClassifier
from torch.utils.data import ConcatDataset, DataLoader

# Helper for timestamped output
def get_output_dir(run_name, base_dir="results", fold_idx=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(base_dir, f"{run_name}_{timestamp}")
    if fold_idx is not None:
        fold_dir = os.path.join(base, f"fold_{fold_idx}")
        logs = os.path.join(fold_dir, "logs"); os.makedirs(logs, exist_ok=True)
        ckpt = os.path.join(fold_dir, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
        results = os.path.join(fold_dir, "results"); os.makedirs(results, exist_ok=True)
    else:
        logs = os.path.join(base, "logs"); os.makedirs(logs, exist_ok=True)
        ckpt = os.path.join(base, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
        results = os.path.join(base, "results"); os.makedirs(results, exist_ok=True)
    return base, logs, ckpt, results, timestamp

def save_hparams_to_dir(hparams, out_dir, timestamp):
    with open(os.path.join(out_dir, f"hparams_{timestamp}.yaml"), "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    with open(os.path.join(out_dir, f"hparams_{timestamp}.json"), "w") as f:
        json.dump(hparams, f, indent=4)

def get_concat_index(dataset, idx):
    if isinstance(dataset, ConcatDataset):
        for i, ds in enumerate(dataset.datasets):
            if idx < len(ds):
                return i, idx
            idx -= len(ds)
    return None, idx

def main():
    parser = argparse.ArgumentParser("Multimodal Multi-Task Training & Evaluation (Joint Pain/Stimulus, Syracuse+BioVid)")
    parser.add_argument("--config", type=str, required=True, help="Path to the unified experiment config YAML")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "cv"], help="Mode: train/test/cv")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for outputs/results")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to load model checkpoint in test mode")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (overrides config if set)")
    parser.add_argument("--strategy", type=str, default="auto", help="Distributed training strategy for PyTorch Lightning Trainer (e.g. 'ddp', 'ddp_spawn', 'auto', 'dp', etc.)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    def validate_config(config):
        # Check for required fields in config
        required_model_fields = ["vae_in_channels", "xdit_in_channels"]
        model_params = config.get("model_params", {})
        missing = [k for k in required_model_fields if k not in model_params]
        if missing:
            raise ValueError(f"Missing required model_params fields in config: {missing}")
        # Add more checks as needed
    validate_config(config)

    run_name = config.get("run_name", "multimodal_multitask_run")
    syracuse_cfg = config.get("syracuse_settings", {})
    biovid_cfg = config.get("biovid_settings", {})
    trainer_cfg = config.get("trainer_params", {})
    batch_size = trainer_cfg.get("batch_size", 32)
    num_workers = trainer_cfg.get("num_workers", 4)
    precision = trainer_cfg.get("precision", "32")
    monitor_metric = trainer_cfg.get("monitor_metric", "val_pain_QWK")
    monitor_mode = trainer_cfg.get("monitor_mode", "max")
    early_stop_patience = trainer_cfg.get("early_stop_patience", 20)
    max_epochs = trainer_cfg.get("max_epochs", 75)
    model_summary_depth = trainer_cfg.get("model_summary_depth", 2)

    # Set up the base directory first (common across all folds in CV mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, f"{run_name}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    save_hparams_to_dir(config, base_dir, timestamp)

    # Use seed from trainer_params in config, fallback to args.seed
    seed_to_use = trainer_cfg.get("seed", args.seed)
    torch.manual_seed(seed_to_use)
    pl.seed_everything(seed_to_use)
    torch.use_deterministic_algorithms(True, warn_only=True)

    import os as _os
    _os.environ["PYTHONHASHSEED"] = str(seed_to_use)

    if args.mode == "cv":
        print("Running 3-fold cross-validation...")
        for fold_idx in range(3):
            print(f"\n=== STARTING FOLD {fold_idx} ===")
            _, logs_dir, ckpt_dir, results_dir, fold_timestamp = get_output_dir(run_name, base_dir, fold_idx)
            # Setup Syracuse multimodal data module for this fold
            syracuse_dm = SyracuseMultimodalDataModule(
                vae_feature_dir=syracuse_cfg["vae_feature_dir"],
                xdit_feature_dir=syracuse_cfg["xdit_feature_dir"],
                meta_path=syracuse_cfg["meta_path"],
                task=syracuse_cfg.get("task", "classification"),
                thresholds=syracuse_cfg.get("thresholds"),
                batch_size=batch_size,
                num_workers=num_workers,
                cv_fold=fold_idx,
                seed=seed_to_use,
                balanced_sampling=syracuse_cfg.get("balanced_sampling", False),
                temporal_pooling=syracuse_cfg.get("temporal_pooling", "none"),
                flatten=syracuse_cfg.get("flatten", False)
            )
            syracuse_dm.setup()
            # For BioVid, use the dataset directly (or a DataModule if you have one)
            biovid_multimodal_ds = BioVidMultimodalDataset(
                vae_feature_dir=biovid_cfg["vae_feature_dir"],
                xdit_feature_dir=biovid_cfg["xdit_feature_dir"],
                meta_path=biovid_cfg["meta_path"],
                temporal_pooling=biovid_cfg.get("temporal_pooling", "none"),
                flatten=biovid_cfg.get("flatten", False)
            )
            wrapped_syracuse = CombinedTaskDatasetWrapper(syracuse_dm.train_dataset, task_name='pain_level', multimodal=True)
            wrapped_biovid = CombinedTaskDatasetWrapper(biovid_multimodal_ds, task_name='stimulus', multimodal=True)
            sample = wrapped_biovid[0]
            train_ds = ConcatDataset([wrapped_syracuse, wrapped_biovid])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            # For validation, use only Syracuse val split
            wrapped_syracuse_val = CombinedTaskDatasetWrapper(syracuse_dm.val_dataset, task_name='pain_level', multimodal=True)
            val_loader = DataLoader(wrapped_syracuse_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            # Model instantiation
            model_cfg = config.get("model_params", {})
            optimizer_cfg = config.get("optimizer_params", {})
            lr_scheduler_cfg = config.get("lr_scheduler_params", {})
            hparams_model = {**model_cfg, **optimizer_cfg}
            if lr_scheduler_cfg.get("use_scheduler", False):
                hparams_model.update({
                    "use_lr_scheduler": True,
                    "lr_factor": lr_scheduler_cfg.get("factor", 0.1),
                    "lr_patience": lr_scheduler_cfg.get("patience", 10),
                    "min_lr": lr_scheduler_cfg.get("min_lr", 1e-6),
                    "monitor_metric": monitor_metric
                })
            else:
                hparams_model["use_lr_scheduler"] = False
            hparams_model["num_pain_classes"] = syracuse_cfg.get("num_classes_pain")
            hparams_model["num_stimulus_classes"] = biovid_cfg.get("num_classes_stimulus")
            model = MultimodalMultiTaskCoralClassifier(**hparams_model)
            # Callbacks/logging
            tb_logger = TensorBoardLogger(logs_dir, name=run_name)
            csv_logger = CSVLogger(logs_dir, name=f"{run_name}_csv", version=fold_timestamp)
            checkpoint_name = f"{run_name}_best_{fold_timestamp}_fold{fold_idx}_seed{seed_to_use}"
            ckpt_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                save_last=True,
                filename=checkpoint_name + "-{epoch}-{" + monitor_metric + ":.3f}",
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=1,
                verbose=True
            )
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=early_stop_patience,
                verbose=True
            )
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            model_summary = ModelSummary(max_depth=model_summary_depth)
            callbacks = [ckpt_callback, early_stop_callback, lr_monitor, model_summary]
            # Add advanced callbacks if enabled in config
            if trainer_cfg.get("use_rich_progress_bar", False):
                callbacks.append(RichProgressBar())
            if trainer_cfg.get("use_swa", False):
                callbacks.append(StochasticWeightAveraging(swa_epoch_start=trainer_cfg.get("swa_epoch_start", 0.8)))
            if trainer_cfg.get("use_gradient_accum", False):
                callbacks.append(GradientAccumulationScheduler(scheduling=trainer_cfg.get("grad_accum_schedule", {0: 1})))
            # Loggers
            loggers = [tb_logger, csv_logger]
            if trainer_cfg.get("use_wandb", False) and WandbLogger is not None:
                loggers.append(WandbLogger(project=trainer_cfg.get("wandb_project", run_name), name=run_name))
            if trainer_cfg.get("use_comet", False) and CometLogger is not None:
                loggers.append(CometLogger(project_name=trainer_cfg.get("comet_project", run_name), experiment_name=run_name))
            trainer = Trainer(
                log_every_n_steps=10,
                devices=args.n_gpus,
                accelerator="cpu" if args.n_gpus == 0 else "gpu",
                max_epochs=max_epochs,
                precision=precision,
                logger=loggers,
                callbacks=callbacks,
                default_root_dir=logs_dir,
                benchmark=True if args.n_gpus > 0 else False,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                enable_model_summary=True,
                deterministic=False,
                #deterministic=True,
                profiler=trainer_cfg.get("profiler", None),
                strategy=args.strategy
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            if val_loader:
                print(f"Fold {fold_idx}: Running test step with best checkpoint on validation data")
                test_results = trainer.test(model=None, dataloaders=val_loader, ckpt_path=ckpt_callback.best_model_path)
                results_filename = os.path.join(results_dir, f"fold_{fold_idx}_results.json")
                with open(results_filename, "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Test results saved to {results_filename}")
    else:
        # Regular train or test mode
        _, logs_dir, ckpt_dir, results_dir, timestamp = get_output_dir(run_name, base_dir)
        syracuse_dm = SyracuseMultimodalDataModule(
            vae_feature_dir=syracuse_cfg["vae_feature_dir"],
            xdit_feature_dir=syracuse_cfg["xdit_feature_dir"],
            meta_path=syracuse_cfg["meta_path"],
            task=syracuse_cfg.get("task", "classification"),
            thresholds=syracuse_cfg.get("thresholds"),
            batch_size=batch_size,
            num_workers=num_workers,
            cv_fold=syracuse_cfg.get("cv_fold", 0),
            seed=seed_to_use,
            balanced_sampling=syracuse_cfg.get("balanced_sampling", False),
            temporal_pooling=syracuse_cfg.get("temporal_pooling", "none"),
            flatten=syracuse_cfg.get("flatten", False)
        )
        syracuse_dm.setup()
        biovid_multimodal_ds = BioVidMultimodalDataset(
            vae_feature_dir=biovid_cfg["vae_feature_dir"],
            xdit_feature_dir=biovid_cfg["xdit_feature_dir"],
            meta_path=biovid_cfg["meta_path"],
            temporal_pooling=biovid_cfg.get("temporal_pooling", "none"),
            flatten=biovid_cfg.get("flatten", False)
        )
        wrapped_syracuse = CombinedTaskDatasetWrapper(syracuse_dm.train_dataset, task_name='pain_level', multimodal=True)
        wrapped_biovid = CombinedTaskDatasetWrapper(biovid_multimodal_ds, task_name='stimulus', multimodal=True)
        sample = wrapped_biovid[0]
        train_ds = ConcatDataset([wrapped_syracuse, wrapped_biovid])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        wrapped_syracuse_val = CombinedTaskDatasetWrapper(syracuse_dm.val_dataset, task_name='pain_level', multimodal=True)
        val_loader = DataLoader(wrapped_syracuse_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # Model instantiation
        model_cfg = config.get("model_params", {})
        optimizer_cfg = config.get("optimizer_params", {})
        lr_scheduler_cfg = config.get("lr_scheduler_params", {})
        hparams_model = {**model_cfg, **optimizer_cfg}
        # Remove 'optimizer' key if present, use 'optimizer_name' instead
        if 'optimizer' in hparams_model:
            hparams_model['optimizer_name'] = hparams_model.pop('optimizer')
        if lr_scheduler_cfg.get("use_lr_scheduler", False):
            hparams_model.update({
                "use_lr_scheduler": True,
                "lr_factor": lr_scheduler_cfg.get("factor", 0.1),
                "lr_patience": lr_scheduler_cfg.get("patience", 10),
                "min_lr": lr_scheduler_cfg.get("min_lr", 1e-6),
                "monitor_metric": monitor_metric
            })
        else:
            hparams_model["use_lr_scheduler"] = False
        hparams_model["num_pain_classes"] = syracuse_cfg.get("num_classes_pain")
        hparams_model["num_stimulus_classes"] = biovid_cfg.get("num_classes_stimulus")
        model = MultimodalMultiTaskCoralClassifier(**hparams_model)
        tb_logger = TensorBoardLogger(logs_dir, name=run_name)
        csv_logger = CSVLogger(logs_dir, name=f"{run_name}_csv", version=timestamp)
        checkpoint_name = f"{run_name}_best_{timestamp}_seed{seed_to_use}"
        ckpt_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_last=True,
            filename=checkpoint_name + "-{epoch}-{" + monitor_metric + ":.3f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            verbose=True
        )
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=early_stop_patience,
            verbose=True
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        model_summary = ModelSummary(max_depth=model_summary_depth)
        callbacks = [ckpt_callback, early_stop_callback, lr_monitor, model_summary]
        # Add advanced callbacks if enabled in config
        if trainer_cfg.get("use_rich_progress_bar", False):
            callbacks.append(RichProgressBar())
        if trainer_cfg.get("use_swa", False):
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=trainer_cfg.get("swa_epoch_start", 0.8)))
        if trainer_cfg.get("use_gradient_accum", False):
            callbacks.append(GradientAccumulationScheduler(scheduling=trainer_cfg.get("grad_accum_schedule", {0: 1})))
        # Loggers
        loggers = [tb_logger, csv_logger]
        if trainer_cfg.get("use_wandb", False) and WandbLogger is not None:
            loggers.append(WandbLogger(project=trainer_cfg.get("wandb_project", run_name), name=run_name))
        if trainer_cfg.get("use_comet", False) and CometLogger is not None:
            loggers.append(CometLogger(project_name=trainer_cfg.get("comet_project", run_name), experiment_name=run_name))
        trainer = Trainer(
            log_every_n_steps=10,
            devices=args.n_gpus,
            accelerator="cpu" if args.n_gpus == 0 else "gpu",
            max_epochs=max_epochs,
            precision=precision,
            logger=loggers,
            callbacks=callbacks,
            default_root_dir=logs_dir,
            benchmark=True if args.n_gpus > 0 else False,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=True,
            profiler=trainer_cfg.get("profiler", None),
            strategy=args.strategy
        )
        if args.mode == "train":
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            print(f"Best checkpoint at: {ckpt_callback.best_model_path}")
        if args.mode in ["test", "train"]:
            if val_loader:
                ckpt_path = args.ckpt_path
                if not ckpt_path and args.mode == "train":
                    ckpt_path = ckpt_callback.best_model_path
                elif not ckpt_path and args.mode == "test":
                    raise ValueError("ckpt_path must be provided for mode=test if not chained after training.")
                print(f"Running test step with checkpoint: {ckpt_path}")
                test_results = trainer.test(model if args.mode == "train" and not ckpt_path else None, dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
                results_filename = os.path.join(results_dir, f"test_results_{timestamp}.json")
                with open(results_filename, "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Test results saved to {results_filename}")
    print(f"All logs, checkpoints, and configs are saved under {base_dir}")

if __name__ == "__main__":
    main() 