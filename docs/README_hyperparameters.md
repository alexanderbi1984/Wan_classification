# Hyperparameter Reference for Multimodal Multi-Task Pipeline

This document explains every hyperparameter in `template_multimodal.yaml` for the multimodal multi-task ordinal classification pipeline. For each parameter, we describe its purpose, type, and allowed options (if applicable).

---

## Top-Level
| Name      | Type   | Description |
|-----------|--------|-------------|
| run_name  | str    | Name for the experiment/run. Used in logging and checkpoint naming. |

---

## model_params
| Name                | Type    | Description | Options/Notes |
|---------------------|---------|-------------|--------------|
| vae_in_channels     | int     | Number of channels for VAE features. | **Required** |
| xdit_in_channels    | int     | Number of channels for xDiT features. | **Required** |
| vae_processor       | str     | Module to process VAE features. | `resnet3d`, `linear_bn`, ... |
| xdit_processor      | str     | Module to process xDiT features. | `linear_bn`, ... |
| fusion_type         | str     | How to fuse VAE and xDiT features. | `cat_proj_bn_relu`, `cat_proj_bn_relu_res`, ... |
| fusion_dim          | int     | Output dim after fusion. | |
| temporal_encoder    | str     | Temporal encoder type. | `gru`, `transformer`, ... |
| temporal_dim        | int     | Output dim of temporal encoder. | |
| pooling             | str     | Temporal pooling method. | `mean`, `max`, `cls` |
| shared_mlp_dim      | int     | Hidden dim for shared MLP after pooling. | |
| dropout             | float   | Dropout rate in fusion/MLP. | 0.0 - 1.0 |
| num_heads           | int     | Number of heads (for transformer). | Only for transformer |

---

## syracuse_settings
| Name              | Type   | Description | Options/Notes |
|-------------------|--------|-------------|--------------|
| meta_path         | str    | Path to Syracuse meta CSV. | **Required** |
| task              | str    | Task type. | `classification` |
| thresholds        | list/None | Custom thresholds for ordinal bins. | Optional |
| num_classes_pain  | int    | Number of pain classes. | **Required** |
| cv_fold           | int    | Cross-validation fold index. | 0, 1, 2 |
| balanced_sampling | bool   | Use balanced sampling. | true/false |
| temporal_pooling  | str    | Pooling for input features. | `none`, `mean`, ... |
| flatten           | bool   | Flatten input features. | true/false |

---

## biovid_settings
| Name                | Type   | Description | Options/Notes |
|---------------------|--------|-------------|--------------|
| features_path       | str    | Path to BioVid features (npy). | **Required** |
| meta_path           | str    | Path to BioVid meta CSV. | **Required** |
| num_classes_stimulus| int    | Number of stimulus classes. | **Required** |
| split_ratio         | float  | Train/val split ratio. | 0.0 - 1.0 |
| temporal_pooling    | str    | Pooling for input features. | `none`, `mean`, ... |
| flatten             | bool   | Flatten input features. | true/false |

---

## trainer_params
| Name                  | Type    | Description | Options/Notes |
|-----------------------|---------|-------------|--------------|
| batch_size            | int     | Batch size for training. | |
| num_workers           | int     | DataLoader worker count. | |
| precision             | int/str | Precision for training. | `32`, `16`, `bf16` |
| max_epochs            | int     | Max number of epochs. | |
| early_stop_patience   | int     | Early stopping patience. | |
| model_summary_depth   | int     | Model summary depth for logging. | |
| monitor_metric        | str     | Metric to monitor for checkpointing. | `val_pain_QWK`, `val_pain_MAE` |
| seed                  | int     | Random seed. | |
| profiler              | str/None| PyTorch Lightning profiler. | `simple`, `advanced`, `None` |
| use_rich_progress_bar | bool    | Enable RichProgressBar callback. | true/false |
| use_swa               | bool    | Enable Stochastic Weight Averaging. | true/false |
| swa_epoch_start       | float   | When to start SWA (fraction of epochs). | 0.0 - 1.0 |
| use_gradient_accum    | bool    | Enable GradientAccumulationScheduler. | true/false |
| grad_accum_schedule   | dict    | Gradient accumulation schedule. | e.g. `{0: 1, 10: 2}` |
| use_wandb             | bool    | Enable WandbLogger. | true/false |
| wandb_project         | str     | Wandb project name. | |
| use_comet             | bool    | Enable CometLogger. | true/false |
| comet_project         | str     | Comet project name. | |

---

## optimizer_params
| Name         | Type   | Description | Options/Notes |
|--------------|--------|-------------|--------------|
| optimizer    | str    | Optimizer type. | `adam`, `sgd`, ... |
| lr           | float  | Learning rate. | |
| weight_decay | float  | Weight decay. | |

---

## lr_scheduler_params
| Name         | Type   | Description | Options/Notes |
|--------------|--------|-------------|--------------|
| use_scheduler| bool   | Enable LR scheduler. | true/false |
| factor       | float  | LR reduction factor. | |
| patience     | int    | LR scheduler patience. | |
| min_lr       | float  | Minimum LR. | |

---

## Notes
- **Required** fields must be set for the pipeline to run.
- For any new model or data options, add them to the YAML and document here.
- For advanced callbacks/logging, ensure the relevant packages (wandb, comet_ml) are installed if enabled. 