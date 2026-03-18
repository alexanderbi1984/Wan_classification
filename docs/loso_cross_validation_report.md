# Leave-One-Subject-Out (LOSO) Cross-Validation Report

**Date**: March 17, 2026  
**Status**: In progress (46/87 folds completed)

---

## 1. Motivation

Our previous fixed-split evaluation (train/val/test) yielded a best test accuracy of ~32.3% and QWK of ~0.43. However, the SOTA paper on BioVid (*A Full Transformer-based Framework for Automatic Pain Estimation using Videos*, Zhong et al.) reports significantly higher results using a **Leave-One-Subject-Out (LOSO)** protocol. LOSO is considered the gold standard for subject-independent evaluation in affective computing because:

- It maximizes training data utilization (86 subjects for training vs. our fixed split's ~60).
- It eliminates the risk of subject-specific shortcuts leaking across splits.
- It provides per-subject generalization metrics with statistical confidence intervals.

## 2. Implementation

### 2.1 Scripts

| File | Purpose |
|------|---------|
| `scripts/train_loso_fold.py` | Trains and evaluates a single LOSO fold for a given test subject |
| `scripts/aggregate_loso.py` | Aggregates results from all 87 folds into summary statistics |
| `slurm/loso_batch1.sh` | SLURM script for subjects 1–29 |
| `slurm/loso_batch2.sh` | SLURM script for subjects 30–58 |
| `slurm/loso_batch3.sh` | SLURM script for subjects 59–87 |

### 2.2 Fold Construction

For each of the 87 subjects in BioVid:
1. **Test set**: All samples from the held-out subject (~100 samples, 20 per class).
2. **Validation set**: 5 randomly selected subjects from the remaining 86 (using a deterministic seed based on test subject ID).
3. **Training set**: All remaining subjects (~81 subjects, ~8,100 samples).

### 2.3 Training Configuration

Each fold uses the best-performing setup from our fixed-split experiments:

| Parameter | Value |
|-----------|-------|
| Config | `config_lora_t100_aug3.yaml` |
| DiT timestep | 100 |
| LoRA rank | 4 |
| Loss | CORAL ordinal |
| Augmentation | TrivialAugmentWide + HorizontalFlip + ColorJitter + RandomGrayscale + RandomErasing |
| Batch size | 20 (reduced from 24 to prevent OOM on specific subjects) |
| Accumulate grad batches | 1 |
| GPUs | 4 × H100 (DDP) |
| Max epochs | 30 |
| Early stopping | patience=20, monitor=val_pain_QWK |
| Monitor metric | val_pain_QWK (maximize) |

### 2.4 Resource Management

- **CPU quota**: The SLURM account has a 48-CPU group limit. Each LOSO batch requests 16 CPUs, allowing 3 concurrent batches (3 × 16 = 48).
- **Checkpoint cleanup**: Each fold deletes its temporary checkpoint directory after extracting test metrics, preventing disk quota exhaustion.
- **Resume capability**: Batch scripts skip subjects that already have result JSON files, enabling interrupted jobs to resume.
- **VRAM management**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to handle memory fragmentation across folds with varying sample counts.

### 2.5 Execution Timeline

| Round | Jobs | Subjects | Status |
|-------|------|----------|--------|
| Round 1 | Initial submission (32 CPUs each) | Batch 1–3 | Failed: AssocGrpCpuLimit |
| Round 2 | Resubmitted (16 CPUs each) | Batch 1–3 | Partially completed, some folds OOM |
| Round 3 | Re-queued with batch_size=20 | Remaining folds | Completed some, re-queued remainder |
| Round 4+ | Continued re-queuing | Remaining folds | Ongoing |

## 3. Results (46/87 Folds)

### 3.1 Aggregate Metrics

| Metric | Mean | Std | Median | 95% CI |
|--------|------|-----|--------|--------|
| Accuracy | 0.2701 | 0.0865 | 0.2400 | [0.2441, 0.2960] |
| QWK | 0.2816 | 0.2557 | 0.2526 | [0.2048, 0.3584] |
| Macro F1 | 0.2023 | 0.1003 | 0.1810 | [0.1722, 0.2324] |
| MAE | 1.0990 | 0.2319 | 1.1450 | [1.0293, 1.1686] |

### 3.2 Per-Class Recall

| Class | Mean Recall | Std | Median |
|-------|------------|-----|--------|
| BL1 (c0) | 0.0065 | 0.0268 | 0.0000 |
| PA1 (c1) | 0.4239 | 0.3238 | 0.4250 |
| PA2 (c2) | 0.4043 | 0.3263 | 0.3000 |
| PA3 (c3) | 0.1641 | 0.1686 | 0.1250 |
| PA4 (c4) | 0.2891 | 0.3217 | 0.2000 |

### 3.3 Comparison with Fixed-Split

| Metric | Fixed-Split Best | LOSO Mean (46 folds) |
|--------|-----------------|---------------------|
| Accuracy | 0.323 | 0.270 |
| QWK | 0.430 | 0.282 |
| MAE | 0.938 | 1.099 |
| BL1 Recall | 0.000 | 0.007 |

## 4. Key Findings

### Finding 1: LOSO Performance is Lower Than Fixed-Split

The LOSO evaluation shows consistently lower performance than our fixed-split results. This is expected because:
- LOSO tests on completely unseen subjects, while the fixed split may have correlated subjects across splits.
- Each fold has a different test distribution (one subject's idiosyncratic expressions).
- High variance across folds (QWK std=0.256) indicates extreme subject dependency.

### Finding 2: BL1 Recall Remains Near Zero

Across 46 folds, the mean BL1 recall is 0.65% (median 0.0%). This is consistent with the fixed-split finding and confirms that the Wan DiT backbone fundamentally cannot distinguish "no pain" from "low pain" facial expressions, regardless of the evaluation protocol.

### Finding 3: High Inter-Subject Variance

The standard deviation of QWK (0.256) is nearly as large as the mean (0.282), indicating that model performance is highly subject-dependent. Some subjects are easily classified (clear pain expressions), while others are nearly impossible (subtle or atypical expressions).

### Finding 4: PA1 and PA2 Have the Highest Recall

PA1 (0.424) and PA2 (0.404) have the best per-class recall, suggesting the model has a bias toward predicting mid-range pain levels. PA3 (0.164) is unexpectedly low, possibly because PA3 samples are often confused with PA2 or PA4.

## 5. Status and Next Steps

- **46/87 folds completed** (52.9%). Remaining folds are being processed in ongoing SLURM jobs.
- Final aggregation will be performed once all 87 folds complete.
- Results will be compared with the SOTA paper's LOSO metrics for a fair cross-study comparison.
