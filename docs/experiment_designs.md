# Experiment Design Summary

This document details all experiments conducted in the Wan_classification project for BioVid pain classification using WanModel 14B with LoRA fine-tuning.

## Common Architecture

All experiments share the following architecture:

- **Backbone**: WanModel 14B (Diffusion Transformer) with LoRA adapters
- **VAE**: WanVAE (3D Causal VAE) for encoding raw video frames into latent representations
- **Feature Extraction**: Dual-stream (VAE features via r3d_18 + DiT features), fused via concatenation and shared encoder
- **Classification Head**: CORAL ordinal head (K-1 cumulative probabilities) and/or CE head (K class logits)
- **Dataset**: BioVid Heat Pain Database, 5-class ordinal pain classification (BL1, PA1, PA2, PA3, PA4)
- **Data Split**: Subject-level split (no subject overlap between train/val/test)
  - Train: ~6000 videos | Val: ~780 videos | Test: ~1220 videos (BL1=180, PA1-PA4=260 each in test)
- **Training Framework**: PyTorch Lightning with DDP (4x GPUs)
- **Precision**: bf16-mixed
- **Gradient Checkpointing**: Enabled
- **Optimizer**: AdamW with CosineAnnealingLR scheduler
- **Early Stopping**: patience=20, monitoring val_pain_QWK (maximize)
- **LoRA Config**: rank=4, alpha=1.0 (applied to DiT attention layers)
- **Learning Rate**: lr_backbone=5e-5, lr_head=1e-3

---

## Experiment Timeline

### Phase 1: Early Prototypes (Batch Size Exploration)

These experiments used small per-GPU batch sizes with gradient accumulation.

#### Exp 1: dim256_v1 (Job 29499064)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256.yaml` (early version) |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 (default) |
| Batch Size / Accumulate | 2 / 4 (effective batch size = 32 on 4 GPUs) |
| Loss | CORAL only |
| Augmentation | aug1 (standard) |
| Prompt Conditioning | No |
| **Rationale** | First experiment with increased feature dimensionality. Small batch size due to initial VRAM uncertainty. |

#### Exp 2: t100_v1 (Job 29499265)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100.yaml` (early version) |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 2 / 4 (effective batch size = 32 on 4 GPUs) |
| Loss | CORAL only |
| Augmentation | aug1 (standard) |
| Prompt Conditioning | No |
| **Rationale** | Test noise injection via non-zero DiT timestep (t=100/1000). Hypothesis: adding slight noise to latents forces DiT to extract more robust, denoising-aware features. |

---

### Phase 2: Batch Size Optimization (batch_size=24)

After VRAM profiling showed ~6 GB headroom (87.8 GB used / 95.6 GB total), batch size was increased from 2 to 24 per GPU and accumulation reduced from 4 to 1, yielding effective batch size = 96. This significantly improved wall-clock training speed.

#### Exp 3: dim256 (Job 29499363)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 (default) |
| Batch Size / Accumulate | 24 / 1 (effective batch size = 96) |
| Loss | CORAL only |
| Augmentation | aug1 (standard) |
| Prompt Conditioning | No |
| **Rationale** | Re-run of dim256_v1 with optimized batch size. |

#### Exp 4: t100 (Job 29499362)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 (effective batch size = 96) |
| Loss | CORAL only |
| Augmentation | aug1 (standard) |
| Prompt Conditioning | No |
| **Rationale** | Re-run of t100_v1 with optimized batch size. |

---

### Phase 3: Stronger Data Augmentation (aug2)

After observing a val-test generalization gap, augmentation was strengthened.

**aug1 (standard) settings:**

| Augmentation | Parameter |
|-------------|-----------|
| TrivialAugment | prob=0.4 |
| Horizontal Flip | prob=0.5 |
| Color Jitter | prob=0.5, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1 |
| Random Grayscale | prob=0.2 |
| Random Erasing | prob=0.3, scale=[0.02, 0.33] |

**aug2 (stronger) settings:**

| Augmentation | Parameter |
|-------------|-----------|
| TrivialAugment | prob=0.6 |
| Horizontal Flip | prob=0.5 |
| Color Jitter | prob=0.8, brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15 |
| Random Grayscale | prob=0.3 |
| Random Erasing | prob=0.4, scale=[0.05, 0.4] |

#### Exp 5: dim256_aug2 (Job 29508806)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_aug2.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | **aug2 (stronger)** |
| Prompt Conditioning | No |
| **Rationale** | Stronger augmentation to reduce val-test gap observed in dim256 (gap=0.056). |

#### Exp 6: t100_aug2 (Job 29508807)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug2.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | **aug2 (stronger)** |
| Prompt Conditioning | No |
| **Rationale** | Stronger augmentation for t100 variant. |

---

### Phase 4: Text Prompt Conditioning

Hypothesis: leveraging DiT's existing cross-attention mechanism with task-descriptive text prompts could improve feature quality, since DiT was pre-trained with text conditioning.

**Design:**
- Pre-computed T5 embeddings for a pool of 15 semantically similar but lexically diverse prompts describing the pain classification task
- During training: randomly sample one prompt embedding per forward pass (acts as augmentation)
- During inference: use a single fixed prompt (first in the pool)
- Embeddings stored in `prompt_embeddings.pt`

**Prompt pool examples:**
- "A close-up video of a person's face showing pain expression."
- "Clinical monitoring of facial responses to thermal pain."
- "The subtle and intense facial manifestations of physical pain."

#### Exp 7: dim256_prompt (Job 29509013)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_prompt.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug2 |
| Prompt Conditioning | **Yes** (15-prompt pool) |
| **Rationale** | Test if task-descriptive text conditioning through DiT cross-attention improves classification. |

#### Exp 8: t100_prompt (Job 29509014)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_prompt.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug2 |
| Prompt Conditioning | **Yes** (15-prompt pool) |
| **Rationale** | Prompt conditioning combined with noise injection. |

---

### Phase 5: Loss Function Exploration

Motivated by persistent class 0 (BL1) recall = 0 under CORAL loss. Investigation of the MMA project revealed it used CE-dominated loss. CORAL's cumulative sigmoid structure creates an ordinal bias that tends to merge the lowest class into adjacent classes.

#### Exp 9: pure CE (Job 29509365)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_ce.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | **CE only** (coral_alpha=0.0, ce_alpha=1.0) |
| Label Smoothing | 0.05 |
| Eval Head | CE (argmax) |
| Augmentation | aug2 |
| Prompt Conditioning | No |
| **Rationale** | Baseline CE loss to test if it resolves class 0 prediction bias. |

#### Exp 10: hybrid (Job 29509366)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_hybrid.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | **Hybrid** (coral_alpha=0.2, ce_alpha=0.8) |
| Label Smoothing | 0.05 |
| Eval Head | CE (argmax) |
| Augmentation | aug2 |
| Prompt Conditioning | No |
| **Rationale** | CE-dominated hybrid to get class balance from CE while retaining ordinal structure from CORAL. Inspired by MMA project's loss design. |

#### Exp 11: CE + Focal Loss (Job 29509371)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_ce_focal.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | **CE + Focal** (coral_alpha=0.0, ce_alpha=1.0, ce_focal_gamma=2.0) |
| Label Smoothing | 0.05 |
| Eval Head | CE (argmax) |
| Augmentation | aug2 |
| Prompt Conditioning | No |
| **Rationale** | Focal loss down-weights easy examples and up-weights hard examples, potentially improving class balance. |

#### Exp 12: hybrid + prompt (Job 29509370)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_dim256_hybrid_prompt.yaml` |
| Feature Dim / Fusion Dim | 256 / 256 |
| DiT Timestep | 0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | **Hybrid** (coral_alpha=0.2, ce_alpha=0.8) |
| Label Smoothing | 0.05 |
| Eval Head | CE (argmax) |
| Augmentation | aug2 |
| Prompt Conditioning | **Yes** (15-prompt pool) |
| **Rationale** | Combination of hybrid loss with prompt conditioning. |

---

## Supplementary: Prompt Ensemble Test (Job 29509369)

- **Script**: `scripts/test_prompt_ensemble.py`
- **Purpose**: Evaluate whether running all 15 prompts at test time and aggregating predictions (soft voting / hard voting) improves over single-prompt inference.
- **Execution**: DDP across 4 GPUs with `torchrun`
- **Checkpoint**: Best checkpoint from dim256_prompt experiment (CORAL-only)
- **Result**: Prompt ensemble did not meaningfully improve over single-prompt inference.

---

## Design Decision Summary

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LoRA rank=4 | Minimal trainable params for downstream task | Effective, stable training |
| batch_size 2 to 24 | VRAM profiling showed headroom | ~3x faster per-epoch, similar or better convergence |
| aug1 to aug2 | Reduce val-test generalization gap | Reduced gap from 0.056 to 0.026 (dim256 series) |
| aug2 to aug3 | Further strengthen regularization | New best QWK=0.407 (t100); dim256 over-regularized |
| MixUp (α=0.4) | De-identification augmentation | **Negative**: hurts QWK with CORAL (-0.03), helps balance |
| DiT t=100 | Noise injection for robust features | Slightly better test QWK, smallest val-test gap |
| dim=256 vs 128 | Higher capacity feature processing | Higher val QWK but larger gap; better with aug2 |
| Prompt conditioning | Leverage DiT cross-attention | **Negative**: consistently worse than no-prompt baselines |
| CORAL to CE/Hybrid | Address class 0 recall = 0 | CE improves class balance but lowers QWK |
| Focal loss (gamma=2) | Focus on hard examples | **Dangerous**: severe test collapse despite good val |
| Binary BL1 diagnostic | Probe BL1 separability per pain level | **Key finding**: BL1 vs PA1 completely inseparable |
| MAE as monitor_metric | Test if MAE-based checkpoint selection helps | **Negative**: strictly worse on ALL test metrics including MAE itself |
| LoRA rank 4→8 | Double adapter capacity | **Negative**: slight degradation (-0.011 QWK), rank=4 sufficient |
| Attention temporal pooling | Learnable frame weighting | **Negative**: overfits quickly (best ep=7), -0.050 QWK vs mean pool |
| Multi-layer DiT features | Combine shallow+deep layers | **Negative**: worst new method (-0.072 QWK), shallow features add noise |
| Test-Time Augmentation | Ensemble over augmented views at inference | **Negative**: augmentation noise hurts (-0.030 QWK) |
| Model Ensemble (3 models) | Combine diverse model predictions | **Marginal**: slight Acc/F1 gain but QWK worse (-0.008) |

---

## Phase 5: Stronger Augmentation (aug3) and MixUp

### Experiment 13: t100_aug3
- **Config**: `config_lora_t100_aug3.yaml`
- **Augmentation**: aug3 (stronger color jitter, higher erasing prob, more aggressive transforms)
- **Early stopping patience**: 10 (reduced from 20 based on convergence analysis)
- **Result**: New best test QWK=0.407, val-test gap=0.023

### Experiment 14: dim256_aug3
- **Config**: `config_lora_dim256_aug3.yaml`
- **Result**: Test QWK=0.362, inferior to dim256_aug2 (0.382) — over-regularization

### Experiment 15: t100_aug3_mixup
- **Config**: `config_lora_t100_aug3_mixup.yaml`
- **MixUp alpha**: 0.4 (blends inputs and labels during training)
- **Goal**: De-identification augmentation to prevent identity shortcuts
- **Result**: Test QWK=0.373 (worse than t100_aug3), but better accuracy (0.301)

### Experiment 16: dim256_aug3_mixup
- **Config**: `config_lora_dim256_aug3_mixup.yaml`
- **Result**: Test QWK=0.330, worst dim256 variant

---

## Phase 6: Binary Classification Diagnostic

Designed to isolate the root cause of BL1 (class 0) recall=0 in the 5-class setting. Each experiment trains a fresh binary classifier for BL1 vs one pain level.

### Shared settings:
- **Loss**: Pure CE (ce_alpha=1.0, coral_alpha=0.0)
- **Dim**: 256, **t**: 0, **Aug**: aug2
- **Label smoothing**: 0.05
- **num_classes**: 2
- **Early stopping patience**: 10

### Experiment 17: BL1 vs PA4 (most distant pair)
- **Config**: `config_lora_binary_bl1_pa4.yaml`, class_subset=[0,4]
- **Result**: Test QWK=0.468, BL1 recall=0.839 — **strong separation**

### Experiment 18: BL1 vs PA3
- **Config**: `config_lora_binary_bl1_pa3.yaml`, class_subset=[0,3]
- **Result**: Test QWK=0.251, BL1 recall=0.850 — **moderate separation**

### Experiment 19: BL1 vs PA2
- **Config**: `config_lora_binary_bl1_pa2.yaml`, class_subset=[0,2]
- **Result**: Test QWK=-0.030, BL1 recall=0.172 — **near random**

### Experiment 20: BL1 vs PA1 (most similar pair)
- **Config**: `config_lora_binary_bl1_pa1.yaml`, class_subset=[0,1]
- **Result**: Test QWK=0.000, BL1 recall=0.000, train loss≈0.693 (random baseline) — **complete failure**
- The model never learned any discriminative features in 10 epochs

---

## Phase 7: Monitor Metric Ablation

### Motivation

All previous experiments used `val_pain_QWK` (maximize) as the early stopping and checkpoint selection criterion. An analysis of existing training logs revealed that `val_pain_QWK` and `val_pain_MAE` peak at different epochs in 7/8 experiments (gap ranges from 1 to 12 epochs), with MAE consistently peaking earlier. This raised the question: does the choice of monitor metric affect test performance?

### Pre-experiment Log Analysis

| Experiment | Best QWK Epoch | Best MAE Epoch | Gap | QWK Loss (if MAE used) |
|---|---|---|---|---|
| t100_aug3 | 14 | 13 | 1 | -0.017 |
| t100_aug2 | 21 | 16 | 5 | -0.002 |
| dim256_aug2 | 22 | 10 | 12 | -0.029 |
| dim256 | 14 | 13 | 1 | -0.029 |
| t100 | 21 | 11 | 10 | -0.020 |
| t100_aug3_mixup | 21 | 16 | 5 | -0.024 |
| pure_CE | 13 | 3 | 10 | -0.057 |
| hybrid | 9 | 5 | 4 | -0.001 |

### Experiment 21: t100_aug3_mixup with MAE Monitor

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug3_mixup_mae_monitor.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug3 |
| MixUp | 0.4 |
| Prompt Conditioning | No |
| Monitor Metric | **val_pain_MAE (minimize)** — only change vs Exp 15 |
| **Rationale** | Controlled ablation: identical to Exp 15 (t100_aug3_mixup) except monitor_metric changed from val_pain_QWK to val_pain_MAE, to measure the impact of checkpoint selection criterion on test performance. |

---

## Phase 8: Architectural & Inference Strategies

Motivated by the performance plateau at Test QWK≈0.407. This phase explores whether changes to LoRA capacity, inference strategies, or temporal/layer feature extraction can push beyond the ceiling.

### Experiment 22: LoRA rank=8 (Job 29591944)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug3_rank8.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug3 |
| MixUp | No |
| LoRA Rank | **8** (doubled from baseline rank=4) |
| LoRA Trainable Params | 13,107,200 (2x baseline) |
| **Rationale** | Test if doubling LoRA rank (and thus adapter capacity) enables the DiT to extract more discriminative features. |

### Experiment 23: Attention-based Temporal Pooling (Job 29591945)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug3_attn_pool.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug3 |
| Temporal Pooling | **attention** (replaces mean pooling) |
| **Rationale** | Learnable attention over temporal dimension to focus on the most discriminative frames. |

### Experiment 24: Multi-layer DiT Feature Extraction (Job 29591946)

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug3_multilayer.yaml` |
| Feature Dim / Fusion Dim | 128 / 128 |
| DiT Timestep | 100.0 |
| Batch Size / Accumulate | 24 / 1 |
| Loss | CORAL only |
| Augmentation | aug3 |
| DiT Feature Layers | **[-1, -10, -20]** (last, 10th-from-last, 20th-from-last of 40 blocks) |
| Feature Combination | Learnable scalar mix (softmax weights + gamma scaling) |
| **Rationale** | ELMo-style combination of shallow, mid, and deep DiT features for richer representations. |

### Experiment 25: Test-Time Augmentation — TTA (Job 29592092)

| Parameter | Value |
|-----------|-------|
| Script | `scripts/test_tta.py` |
| Checkpoint | t100_aug3 best (epoch 14, val QWK=0.430) |
| n_aug | 10 (10 augmented forward passes per sample) |
| Augmentation | aug3 config (same as training) |
| Aggregation | Soft voting (average probabilities across augmented views) |
| GPU | Single H100, batch_size=24 |
| **Rationale** | Ensemble over augmented test-time views to reduce prediction variance. |

### Experiment 26: Model Ensemble (Job 29592114)

| Parameter | Value |
|-----------|-------|
| Script | `scripts/test_ensemble.py` |
| Models | 3 independently trained models: |
| | 1. t100_aug3 (val QWK=0.430) — CORAL, dim=128, t=100 |
| | 2. t100_aug2 (val QWK=0.414) — CORAL, dim=128, t=100 |
| | 3. dim256_aug2 (val QWK=0.408) — CORAL, dim=256, t=0 |
| Aggregation | Soft voting (average probabilities from CORAL heads) |
| Loading Strategy | Sequential (load/infer/unload one model at a time) |
| GPU | Single H100, batch_size=24 |
| **Rationale** | Combine diverse models with different configurations to capture complementary patterns. |
