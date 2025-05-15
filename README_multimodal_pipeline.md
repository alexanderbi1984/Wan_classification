# Multimodal Multi-Task Ordinal Classification Pipeline

## Overview
This pipeline supports multimodal, multi-task ordinal classification for pain and stimulus prediction, fusing VAE and xDiT features. It is modular, extensible, and fully configurable via YAML.

## Configuration
All experiment settings are controlled via a single YAML config file. A template is provided at:

```
Wan_classification/config_pain/template_multimodal.yaml
```

### Config Sections
- **run_name**: Name for the experiment/run.
- **model_params**: Model architecture, fusion, and feature processing settings. Required fields: `vae_in_channels`, `xdit_in_channels`.
- **syracuse_settings**: Syracuse dataset paths and options.
- **biovid_settings**: BioVid dataset paths and options.
- **trainer_params**: Training, logging, and callback settings (batch size, precision, early stopping, advanced callbacks/loggers, etc.).
- **optimizer_params**: Optimizer and learning rate.
- **lr_scheduler_params**: Learning rate scheduler options.

Each section is documented in the template YAML with comments.

### Advanced Callbacks & Logging
You can enable advanced features in `trainer_params`:
- `use_rich_progress_bar`: Show ETA and throughput.
- `use_swa`: Enable Stochastic Weight Averaging.
- `use_gradient_accum`: Use Gradient Accumulation Scheduler.
- `use_wandb`/`use_comet`: Enable Wandb or Comet logging for rich experiment tracking.
- `profiler`: Set to `simple` or `advanced` for bottleneck profiling.

## Data Handling
The pipeline uses a single `MultimodalDataModule` that wraps both Syracuse and BioVid datasets, ensuring clean and reproducible data loading for all modes (train/test/cv).

## Reproducibility
- All random seeds are set (including `PYTHONHASHSEED`).
- Trainer runs in deterministic mode for full reproducibility.
- Checkpoint filenames include fold and seed for clarity.

## Usage Example
```bash
python train_multimodal_multitask.py --config config_pain/template_multimodal.yaml --mode train --n_gpus 1
```

## Extending the Pipeline
- Add new model/fusion/temporal encoder options in `model_params`.
- Add new datasets by extending the DataModule.
- All new features should be exposed via the YAML config for consistency.

See the template YAML for all available options and required fields. 