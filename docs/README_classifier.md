# Classifier Folder

This folder contains modules and configs for PyTorch Lightning-based ordinal and multi-task classifiers.

## Contents
- **multi_task_coral.py**: Main implementation of `MultiTaskCoralClassifier`, a flexible multi-task model supporting CORAL for ordinal regression in pain/stimulus multi-task settings. Includes robust, configurable architectures, metric logging, and advanced loss options.
- **config_multi_task_coral.yaml**: Structured sample YAML for controlling all model, optimizer, and training settings. Add new configs here to manage experiment variants (e.g., with/without LoRA, different heads, etc).

## Multi-Task CORAL Classifier

### Architecture Overview
The `MultiTaskCoralClassifier` implements a multi-task learning approach for ordinal classification with the following components:

1. **Shared Encoder**: A configurable MLP network that processes the input features
   - Supports variable hidden dimensions via `encoder_hidden_dims`
   - Configurable dropout rate via `encoder_dropout`

2. **Dual Task-Specific Heads**: Two separate CORAL (Consistent Rank Logits) heads
   - Pain classification head
   - Stimulus classification head 
   - Each head outputs logits for (num_classes - 1) binary classification tasks

3. **Advanced Loss Functions**:
   - CORAL ordinal regression loss for both tasks
   - Support for label smoothing, focal weighting, and distance penalties
   - Task weighting via `pain_loss_weight` and `stim_loss_weight`
   - Handles class imbalance with optional class weights

4. **Comprehensive Metrics Tracking**:
   - Mean Absolute Error (MAE)
   - Quadratic Weighted Kappa (QWK)
   - Accuracy
   - Confusion matrices for detailed analysis

### CORAL Algorithm

The CORAL (Consistent Rank Logits) algorithm transforms ordinal regression into a series of binary classifications:

1. For a task with K classes (0 to K-1), the model predicts K-1 binary thresholds
2. Each threshold i determines if the class is > i
3. The final class prediction is the sum of positive threshold predictions
4. This approach maintains the ordinal relationship between classes

The implementation includes advanced extensions to the basic CORAL approach:
- Distance penalties that increase loss for predictions further from ground truth
- Focal weighting for handling class imbalance
- Label smoothing for better generalization

## Training Pipeline

The `train_multitask.py` script provides a comprehensive training pipeline:

### Key Features
- **Multi-Task Learning**: Simultaneously trains on both Syracuse pain data and BioVid stimulus data
- **Cross-Validation**: Supports 3-fold CV with detailed metrics aggregation
- **Advanced Callbacks**: Early stopping, learning rate scheduling, and model checkpointing
- **Flexible Data Processing**: Supports temporal pooling and flattening operations
- **Unified Configuration**: Single YAML file controls all aspects of training

### Data Flow
1. Loads and processes Syracuse and BioVid datasets separately
2. Wraps each dataset using `CombinedTaskDatasetWrapper` to standardize data format
3. Combines datasets into a single training stream
4. Handles missing labels gracefully during training
5. Supports separate validation on Syracuse pain data

## Usage

1. **Create or Edit a YAML config:**
   - Configure the full pipeline in `config_pain/config_multi_task_coral.yaml`:
    ```yaml
   run_name: "multitask_joint"
   
   syracuse_settings:
     meta_path: "/path/to/meta.json"
     task: "classification"
     thresholds: [1.0, 3.0, 5.0, 7.0]
     num_classes_pain: 5
   
   biovid_settings:
     features_path: "/path/to/features/"
     meta_path: "/path/to/meta.json" 
     num_classes_stimulus: 5
   
   model_params:
      encoder_hidden_dims: [512, 256]
      encoder_dropout: 0.5
     pain_loss_weight: 1.0
     stim_loss_weight: 1.0
   
   optimizer_params:
     optimizer_type: "AdamW"
     learning_rate: 5.0e-5
     weight_decay: 0.01
   
   # ... additional settings
    ```

2. **Run Training:**
   ```bash
   # Regular training
   python train_multitask.py --config config_pain/config_multi_task_coral.yaml --mode train --n_gpus 1
   
   # Cross-validation
   python train_multitask.py --config config_pain/config_multi_task_coral.yaml --mode cv --n_gpus 1
   
   # Testing with a specific checkpoint
   python train_multitask.py --config config_pain/config_multi_task_coral.yaml --mode test --ckpt_path /path/to/checkpoint.ckpt
    ```

3. **Extending the Model:**
   - To add features like LoRA, modify the `MultiTaskCoralClassifier` class
   - Add new parameters in your config YAML
   - Update the model initialization in `train_multitask.py`

## Advanced Options

- **Learning Rate Scheduling**: Enable and configure the ReduceLROnPlateau scheduler
  ```yaml
  lr_scheduler_params:
    use_scheduler: true
    factor: 0.5
    patience: 5
    min_lr: 1.0e-6
  ```

- **Class Imbalance Handling**: Multiple approaches available
  - Balanced sampling via `balanced_sampling: true` in Syracuse settings
  - Class weights (can be calculated dynamically)
  - Focal loss weighting via `focal_gamma` parameter

- **Distance Penalty**: Penalizes predictions further from ground truth
  ```yaml
  model_params:
    use_distance_penalty: true
  ```

## Key Dependencies
- pytorch-lightning
- torchmetrics
- numpy
- pyyaml

See module and config docstrings for further usage notes and details. 