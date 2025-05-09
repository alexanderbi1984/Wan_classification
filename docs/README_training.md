# Pain Classification Training Script

This document provides information about the training script (`train_multitask.py`) for pain classification, including the cross-validation workflow and advanced callback options.

## Overview

The training script supports three modes of operation:
- `train`: Train a model and evaluate it on the validation set
- `test`: Load an existing model checkpoint and evaluate it on the validation set
- `cv`: Perform a full 3-fold cross-validation, training and evaluating on each fold

## Running Cross-Validation

The cross-validation mode trains three separate models, one for each fold of the Syracuse dataset, and reports aggregated metrics. This helps assess model performance more robustly.

Example usage:
```bash
python train_multitask.py --config your_config.yaml --mode cv --monitor_metric val_pain_QWK
```

When running in CV mode, the script:
1. Creates a common root directory for the entire run
2. For each fold (0, 1, 2):
   - Creates fold-specific directories for logs, checkpoints, and results
   - Trains a model using the training portion of that fold
   - Evaluates the model on the validation portion of that fold
   - Saves fold-specific results
3. Aggregates results across all folds and reports mean and standard deviation
4. Saves detailed aggregated results to a JSON file

## Advanced Features

### Learning Rate Scheduling

The model integrates PyTorch's `ReduceLROnPlateau` learning rate scheduler through the Lightning model's `configure_optimizers` method. This reduces the learning rate when performance plateaus, helping models converge better, especially in later stages of training.

To enable the learning rate scheduler, add the following to your configuration:

```yaml
lr_scheduler:
  use_scheduler: true
  factor: 0.5          # Factor to reduce learning rate by
  patience: 5          # Number of epochs with no improvement before reducing LR
  min_lr: 1.0e-6       # Minimum learning rate
```

The scheduler will automatically monitor your chosen metric (`val_pain_QWK` by default) and adjust the learning rate accordingly.

### Progress Bar Configuration

You can configure the progress bar refresh rate in your configuration:

```yaml
progress_bar_refresh_rate: 1  # Refresh rate in batches
```

### Callbacks

The training script supports several callbacks to enhance the training process:

#### CSV Logging

In addition to TensorBoard logging, the script now also logs metrics to CSV files, making it easier to analyze and plot results using external tools like Excel, pandas, or matplotlib.

CSV logs are saved in the logs directory alongside TensorBoard logs.

#### Model Summary 

The script uses PyTorch Lightning's `ModelSummary` callback to print a summary of the model architecture at the start of training. You can configure the depth of the summary:

```yaml
model_summary_depth: 3  # Depth for model summary (default: 2)
```

#### Learning Rate Monitoring

The script monitors and logs learning rates using the `LearningRateMonitor` callback, which helps track learning rate changes when using schedulers.

## Configuration

See `example_config_with_callbacks.yaml` for a complete example configuration including all new callback options.

## Results and Outputs

When running in CV mode, the following outputs are generated:
- `<run_name>_<timestamp>/` - Root directory for the entire run
  - `fold_0/`, `fold_1/`, `fold_2/` - Fold-specific directories
    - `logs/` - TensorBoard and CSV logs
    - `checkpoints/` - Model checkpoints 
    - `results/` - Fold-specific test results
  - `cv_aggregated_results.json` - Aggregated results across all folds
  - `hparams_<timestamp>.yaml/json` - Configuration used for the run 