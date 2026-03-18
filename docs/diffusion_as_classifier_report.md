# Diffusion-as-Classifier (DaC): Technical Report

**Date**: March 17, 2026  
**Status**: Concluded — both vanilla DaC and CFG-enhanced DaC experiments completed

---

## 1. Introduction and Motivation

### 1.1 The Core Problem

Our Grad-CAM analysis (see `gradcam_analysis_report.md`) revealed that the current LoRA + classifier head pipeline suffers from a **spatial information bottleneck**:

```
128×128 face → VAE(8×) → 16×16 → DiT(patch 2×) → 8×8 → XDiTProcessor(mean) → 1×1
```

The `mean(dim=[H,W])` operation in `XDiTFeatureProcessor` discards all spatial information, reducing the face to a single vector. Micro-expressions critical for distinguishing BL1 from PA1 are lost entirely.

### 1.2 Key Insight: The Diffusion Model Already "Sees" the Details

While the **extracted features** lose spatial information, the diffusion model's **denoising process** operates at full latent resolution (16×16). When the model predicts noise/flow at each spatial position, it must understand what occupies that position — including subtle facial features.

**Diffusion-as-Classifier exploits this**: instead of extracting compressed features, we measure how well the model can reconstruct the original video when conditioned on different pain-level descriptions. The reconstruction error is computed pixel-by-pixel in latent space, preserving full spatial resolution.

### 1.3 Theoretical Foundation

Based on the connection between diffusion models and Bayesian classification:

```
P(class | x) ∝ P(x | class) × P(class)
```

For a diffusion model, `P(x | class)` can be estimated via the conditional denoising score:

```
P(class = k | x) ∝ exp(-E[||ε - ε_θ(x_t, t, prompt_k)||²])
```

where `ε_θ` is the noise/flow predicted by the model conditioned on class-specific text prompt `prompt_k`. The class whose prompt yields the lowest prediction error is most consistent with the input.

**References**:
- Li et al., "Your Diffusion Model is Secretly a Zero-Shot Classifier" (ICLR 2024)
- Clark & Jaini, "Text-to-Image Diffusion Models are Zero-Shot Classifiers" (NeurIPS 2024)

## 2. Method

### 2.1 Flow Matching Formulation

Wan uses a **flow matching** objective (not standard DDPM ε-prediction):

```
x_t = (1 - σ) × x_0 + σ × noise        # Forward process (noise injection)
v_pred = DiT(x_t, t, text_context)       # Model predicts flow field
x_0_pred = x_t - σ × v_pred             # Recover clean signal
```

Where:
- `σ` (sigma) controls the noise level (0 = clean, 1 = pure noise)
- `t = σ × 1000` is the timestep passed to the model's sinusoidal embedding
- The model predicts the "velocity" `v = noise - x_0`

### 2.2 Noise Schedule

Wan employs a **shifted flow matching schedule** with `shift = 5.0`:

```python
alphas = np.linspace(1, 1/1000, 1000)[::-1]   # α from 0.001 to 1.0
sigmas_raw = 1.0 - alphas                       # σ from 0.999 to 0.0
sigmas = 5 × sigmas_raw / (1 + 4 × sigmas_raw)  # Shifted schedule
```

The shift biases the schedule toward higher noise levels, which is important for video generation quality but means the model has more experience denoising heavily corrupted inputs.

### 2.3 Classification Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  For each test video x:                                              │
│                                                                      │
│  1. Encode via VAE:  x_0 = VAE.encode(video)  →  (16, T', 16, 16) │
│                                                                      │
│  2. For each timestep t ∈ {200, 500, 800}:                          │
│     a. Sample noise:  ε ~ N(0, I)                                   │
│     b. Add noise:     x_t = (1-σ)·x_0 + σ·ε     where σ = t/1000  │
│                                                                      │
│     c. For each class k ∈ {BL1, PA1, PA2, PA3, PA4}:               │
│        - context_k = T5_embedding(pain_prompt_k)                     │
│        - v_pred = DiT(x_t, t, context_k)                           │
│        - x_0_pred = x_t - σ · v_pred                               │
│        - L_k += MSE(x_0_pred, x_0)                                 │
│                                                                      │
│  3. Classify:  pred = argmin_k(L_k)                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.4 Text Prompts

Each pain class is associated with 3 semantically varied prompts, pre-encoded via UMT5-XXL and averaged into a single embedding per class:

| Class | Example Prompts |
|-------|----------------|
| BL1 | "A video of a person's face with a neutral expression, showing no pain." |
| PA1 | "A video showing a face with very mild pain, slight discomfort." |
| PA2 | "A video showing moderate pain expression on a person's face." |
| PA3 | "A video showing strong pain expression with pronounced grimacing." |
| PA4 | "A video showing extreme pain with severe facial contortion." |

### 2.5 Timestep Selection Rationale

Three timesteps are selected to probe different noise levels:

| Timestep | σ (sigma) | Noise Level | Probing What |
|----------|-----------|-------------|-------------|
| t=200 | 0.20 | 20% noise | Fine-grained details (texture, micro-expressions) |
| t=500 | 0.50 | 50% noise | Mid-level structure (facial pose, expression shape) |
| t=800 | 0.80 | 80% noise | Global semantics (overall face configuration) |

- **Low noise** (t=200): The model sees mostly clean input. Small differences in denoising correspond to subtle feature recognition. Best for BL1 vs PA1 discrimination.
- **High noise** (t=800): The model must rely heavily on the text prompt to guide reconstruction. Tests whether the prompt encodes meaningful class information.

### 2.6 Batching Strategy

All 5 class prompts are evaluated simultaneously as a single batch (B=5):
- The same noisy `x_t` is replicated 5 times
- Each copy is paired with a different class-specific text embedding
- Single batched forward pass → 5 reconstruction errors

This is 5× more efficient than sequential evaluation and fits within H100's 96GB VRAM.

## 3. Implementation Details

### 3.1 Script: `scripts/diffusion_classifier.py`

```
Arguments:
  --checkpoint_dir   Path to Wan2.1-T2V-14B (DiT + VAE + T5)
  --config           Data config YAML (for frames_root, labels_csv)
  --output_dir       Results output directory
  --timesteps        Timestep values for noise injection (default: 200 500 800)
  --n_noise_samples  Noise samples per timestep (default: 3)
  --split            Dataset split (default: test)
  --max_samples      Limit for quick testing
```

### 3.2 Model Loading and Memory Management

```python
# 1. Load all three components
dit_model = WanModel.from_pretrained(checkpoint_dir)   # ~28 GB (bf16 weights)
vae = WanVAE(vae_pth=...)                              # ~1 GB
t5_encoder = T5EncoderModel(...)                        # ~10 GB

# 2. Pre-compute text embeddings for all classes
class_embeddings = precompute_class_embeddings(t5_encoder, ...)

# 3. Unload T5 to free VRAM (no longer needed after pre-computation)
t5_encoder.model.cpu()
del t5_encoder
torch.cuda.empty_cache()
# After: ~29 GB VRAM (DiT + VAE only) — leaves ~67 GB for activations
```

### 3.3 Mixed Precision

The forward pass is wrapped in `torch.amp.autocast("cuda", dtype=torch.bfloat16)`:
- Model parameters (from safetensors) are already in bfloat16
- The model's internal `autocast(dtype=torch.float32)` for time embeddings ensures numerical stability where needed
- Autocast handles the dtype transitions automatically

This gives ~15× speedup over naive float32 execution:
- **float32**: ~0.017 samples/s (750 min ETA)
- **bfloat16 autocast**: ~0.5 samples/s (41 min ETA)

### 3.4 Output Format

Results are saved to `results/dac/dac_results.json`:

```json
{
  "method": "Diffusion-as-Classifier",
  "timesteps": [200, 500, 800],
  "n_noise_samples": 1,
  "metrics": {
    "accuracy": ...,
    "macro_f1": ...,
    "qwk": ...,
    "mae": ...
  },
  "per_class_recall": { "BL1": ..., "PA1": ..., ... },
  "confusion_matrix": [[...]],
  "predictions": [
    {
      "video_id": "...",
      "true": 0,
      "pred": 3,
      "losses": { "0": 0.123, "1": 0.145, ... }
    }
  ]
}
```

The per-sample `losses` field enables post-hoc analysis of how reconstruction errors vary across classes.

## 4. Why This Approach May Succeed Where Feature Extraction Failed

### 4.1 Full Spatial Resolution

| | Feature Extraction (current) | Diffusion-as-Classifier |
|--|---|---|
| Spatial resolution | 4×4 → mean → 1×1 | 16×16 latent (full) |
| Info per position | Compressed to single vector | Per-position reconstruction error |
| Micro-expression sensitivity | None (below resolution floor) | 1 latent pixel ≈ 8×8 input pixels |

### 4.2 Pre-training Knowledge Utilization

The DiT was pre-trained on millions of videos to predict flow fields. This pre-training implicitly encodes:
- **What a face looks like** (facial structure prior)
- **How expressions change** (temporal dynamics of facial movements)
- **Text-visual correspondence** (via cross-attention with T5 embeddings)

Feature extraction only uses the intermediate hidden states, discarding the denoising capability. DaC uses the **full denoising pipeline** — the exact capability the model was optimized for.

### 4.3 Text Conditioning as Class Signal

During feature extraction, text conditioning was optional (dummy zeros worked fine). In DaC, text conditioning is **the class signal**:
- The model denoises differently based on what it's told the content is
- If told "severe pain," it will try to reconstruct strong grimacing features
- If the input actually shows no pain, the "severe pain" reconstruction will have high error
- The mismatch between prompt and reality creates the discriminative signal

### 4.4 Zero-Shot Capability

No training or fine-tuning is required — DaC uses only the pre-trained model weights. This means:
- No risk of overfitting to training set subjects
- No class-imbalance issues (no classifier weights to bias)
- Potentially better generalization to unseen subjects
- Evaluation can be run immediately on new data

## 5. Potential Limitations and Mitigations

### 5.1 Prompt Sensitivity

The classification quality depends on how well the text prompts describe each pain level. Mitigations:
- Use multiple diverse prompts per class (3 in current implementation)
- Average embeddings to smooth out prompt-specific biases
- Future: optimize prompts via prompt engineering or learned prompt tokens

### 5.2 Computational Cost

Each sample requires `n_timesteps × n_noise_samples × 1 forward pass (B=5)`:
- Current: 3 × 1 = 3 batched forward passes per sample
- At 0.5 samples/s for 1220 test samples: ~41 minutes
- With n_noise_samples=3: ~2 hours
- Feasible for evaluation but not for real-time deployment

### 5.3 Noise Sampling Variance

With n_noise_samples=1, classification is influenced by the specific noise realization. Increasing n_noise_samples to 3–5 reduces variance but increases compute proportionally.

### 5.4 Pre-trained vs Fine-tuned Model

The current implementation uses the **vanilla pre-trained** Wan DiT (not LoRA-finetuned). Options:
- **Pre-trained (current)**: True zero-shot, no domain adaptation
- **LoRA-finetuned**: Has seen BioVid data, but was trained at t=100 only; may not generalize to other timesteps
- Future: fine-tune with DaC objective directly (train a classifier-free guidance model)

## 6. Mathematical Details

### 6.1 Flow Matching Forward Process

Given clean latent `x_0` and Gaussian noise `ε ~ N(0, I)`:

```
x_t = α_t · x_0 + σ_t · ε
```

where `α_t = 1 - σ_t` (flow matching interpolation).

### 6.2 Model Prediction

The DiT predicts the velocity field `v_θ`:

```
v_θ(x_t, t, c) ≈ ε - x_0
```

### 6.3 Clean Signal Recovery

```
x̂_0 = x_t - σ_t · v_θ(x_t, t, c)
```

### 6.4 Classification Loss

For class `k` with text context `c_k`:

```
L_k = (1/N) Σ_t Σ_s ||x̂_0^{(k,t,s)} - x_0||²

where:
  t ∈ {t_1, ..., t_T}  (timesteps)
  s ∈ {1, ..., S}       (noise samples)
  x̂_0^{(k,t,s)} = x_t^{(s)} - σ_t · v_θ(x_t^{(s)}, t, c_k)
```

### 6.5 Final Prediction

```
ŷ = argmin_k L_k
```

The class whose text description is most consistent with the input video (lowest reconstruction error) is selected.

### 6.6 Connection to Variational Lower Bound

The MSE reconstruction error is related to the negative log-likelihood:

```
-log P_θ(x_0 | c_k) ≈ E_{t,ε} [||ε - ε_θ(x_t, t, c_k)||²]
                     ≈ E_{t,ε} [σ_t^{-2} ||x̂_0 - x_0||²]
```

By choosing `argmin_k L_k`, we are approximately choosing `argmax_k P_θ(x_0 | c_k)`, which by Bayes' theorem (with uniform prior) equals `argmax_k P(class=k | x_0)`.

## 7. Final Results

**Job**: 29643954 | **Runtime**: 41.7 min | **Config**: t=[200,500,800], n_noise_samples=1

### 7.1 Overall Metrics

| Metric | DaC (zero-shot) | LoRA Classifier (best) | Random Baseline |
|--------|-----------------|----------------------|-----------------|
| Accuracy | 0.174 | 0.323 | 0.200 |
| Macro F1 | 0.132 | 0.263 | ~0.200 |
| QWK | **-0.005** | 0.430 | 0.000 |
| MAE | 1.912 | 0.938 | ~2.0 |

**QWK = -0.005 ≈ 0**: The model performs at exactly random chance level.

### 7.2 Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| BL1 | 0.145 | **0.694** | 0.239 | 180 |
| PA1 | 0.253 | 0.073 | 0.113 | 260 |
| PA2 | 0.000 | **0.000** | 0.000 | 260 |
| PA3 | 0.223 | 0.212 | 0.217 | 260 |
| PA4 | 0.464 | 0.050 | 0.090 | 260 |

### 7.3 Prediction Distribution Collapse

```
Predicted: BL1=865 (71%)  PA1=75 (6%)  PA2=5 (0.4%)  PA3=247 (20%)  PA4=28 (2%)
Actual:    BL1=180 (15%)  PA1=260      PA2=260        PA3=260        PA4=260
```

The model predicts BL1 for 71% of all inputs — a massive distribution collapse.

### 7.4 Confusion Matrix

```
           BL1    PA1    PA2    PA3    PA4
    BL1    125     13      0     40      2
    PA1    176     19      0     58      7
    PA2    195     14      0     49      2
    PA3    182     15      4     55      4
    PA4    187     14      1     45     13
```

### 7.5 Failure Analysis

**The core assumption of DaC fails for this task.** The method assumes that the reconstruction error `||x0_pred - x0||²` is lower when the text prompt matches the actual content. In practice, the opposite occurs:

1. **"No pain" prompt always wins**: A neutral face is geometrically simpler (smooth, symmetric) than a grimacing face. The denoising model naturally produces lower reconstruction error for simpler targets, regardless of the actual input.

2. **Reconstruction error reflects facial complexity, not content match**: The error is dominated by how "easy" the prompt-conditioned target is to reconstruct, not by how well it matches the input.

3. **Ironic reversal**: BL1 recall jumps to 69.4% (from 0% in LoRA classifier) because the model now ALWAYS predicts BL1 — not because it learned to distinguish it.

4. **PA2 complete failure**: Zero predictions for PA2, suggesting "moderate pain" prompts occupy an uninformative middle ground in the model's text-conditional denoising space.

## 8. Follow-up Experiment: CFG-Enhanced DaC

### 8.1 Motivation

The vanilla DaC failure analysis (Section 7.5) identified the root cause as **weak text conditioning**: the 5 class prompts produced reconstruction losses that differed by only ~7%, with BL1 systematically winning due to geometric simplicity rather than semantic match.

**Hypothesis**: Classifier-Free Guidance (CFG) can amplify the text conditioning signal, forcing the model to be more "creative" and prompt-responsive. Wan's generation pipeline already uses CFG (default `guide_scale=5.0`), confirming the model supports unconditional denoising.

### 8.2 CFG-DaC Formulation

```
v_uncond = DiT(x_t, t, ∅)                           # Unconditional (null prompt)
v_cond   = DiT(x_t, t, prompt_k)                    # Conditional (class prompt)
v_guided = v_uncond + w × (v_cond - v_uncond)        # Amplify text signal
x_0_pred = x_t - σ × v_guided
```

When `w = 1.0`, this reduces to vanilla DaC. When `w > 1`, the text conditioning effect is amplified: the model "hallucinates" content aligned with the prompt. If the prompt matches the input, hallucination aligns with reality → low error. If it doesn't, conflict → high error.

### 8.3 Three Scoring Strategies Tested

| Strategy | Formula | Rationale |
|----------|---------|-----------|
| `cfg_mse` | MSE(x_0_guided, x_0) | Reconstruction error with guided denoising |
| `relative` | MSE(x_0_cond, x_0) − MSE(x_0_uncond, x_0) | Marginal improvement from text conditioning |
| `score_norm` | −\|\|v_cond − v_uncond\|\|² | How strongly each prompt "pulls" the denoising |

### 8.4 Computational Efficiency

All 15 combinations (5 guidance scales × 3 scoring methods) share the same forward passes: 1 unconditional + 5 conditional per timestep. Only 20% overhead vs vanilla DaC.

### 8.5 Results

**Job**: 29643988 | **Runtime**: ~51 min | **Config**: t=[200,500,800], w=[1.0, 3.0, 5.0, 7.5, 10.0]

#### 8.5.1 cfg_mse Results Across Guidance Scales

| w | Accuracy | QWK | BL1 preds | PA1 | PA2 | PA3 | PA4 |
|---|----------|-----|-----------|-----|-----|-----|-----|
| 1.0 | 0.151 | -0.010 | **858 (70%)** | 74 | 3 | 245 | 40 |
| 3.0 | **0.221** | **+0.011** | 33 (3%) | 371 | 77 | 580 | 159 |
| 5.0 | 0.224 | -0.006 | 5 (0%) | 403 | 96 | 476 | 240 |
| 7.5 | 0.212 | -0.008 | 2 (0%) | 415 | 105 | 408 | 290 |
| 10.0 | 0.212 | -0.015 | 2 (0%) | 414 | 112 | 378 | 314 |

True distribution: BL1=180 (15%), PA1=PA2=PA3=PA4=260 (21% each)

#### 8.5.2 relative and score_norm — Invariant to Guidance Scale

`relative` produced **identical results at all w values** (Acc=0.151, QWK=-0.010, BL1=858), because the formula `MSE(x_0_cond, x_0) − MSE(x_0_uncond, x_0)` does not involve the guidance scale parameter. Subtracting the unconditional baseline does not remove the BL1 bias because the bias exists in both conditional and unconditional predictions.

`score_norm` predicted BL1 for 99.8% of samples across all w. The BL1 prompt (18 T5 tokens) naturally produces the largest `||v_cond − v_uncond||` due to its longer embedding, not due to semantic match.

### 8.6 Key Findings

**Finding 1: CFG successfully breaks the BL1 distribution collapse.**

At w≥3.0, BL1 predictions dropped from 858 (70%) to 33 (3%). The guidance amplification overcame the "neutral face = easy to reconstruct" bias. This confirms the theoretical mechanism: CFG forces the model to be more prompt-responsive.

**Finding 2: New biases emerge — BL1 collapse becomes PA3 collapse.**

Instead of collapsing on BL1, predictions shifted to PA1/PA3 dominance. At w=3.0, PA3 accounts for 48% of predictions. The "strong pain / grimacing" prompt appears to create a new "easy reconstruction" attractor in the guided space.

**Finding 3: Accuracy remains at random chance regardless of strategy.**

Best result (w=3.0/cfg_mse): Accuracy=0.221, QWK=+0.011. Random baseline: Accuracy=0.200, QWK=0.000. The improvement is negligible and QWK ≈ 0 confirms zero ordinal correlation.

**Finding 4: The problem is not the scoring strategy — it's the model's semantic gap.**

We tested 15 combinations spanning the full design space of single-step DaC. None achieved meaningful classification. This rules out scoring-level fixes and points to a fundamental limitation: the T2V model lacks fine-grained text-visual understanding for pain expressions.

### 8.7 Diagnosis: Why CFG Cannot Save DaC for Pain Classification

The per-prompt average reconstruction loss (across all 1220 samples) from the vanilla experiment reveals the core issue:

```
BL1 prompt: 0.02528     ← Lowest (model finds "no pain" easiest to reconstruct)
PA3 prompt: 0.02595
PA1 prompt: 0.02627
PA4 prompt: 0.02656
PA2 prompt: 0.02664     ← Highest
```

The discrimination ratio (max−min / mean) is only **7%**. This means 93% of the reconstruction error is shared across all prompts (capturing generic face structure), and only 7% varies with the text condition. This tiny signal is dominated by prompt complexity rather than semantic content matching.

CFG amplifies the 7% text-dependent component, but it amplifies **the wrong signal**: the model responds differently to different prompts, but not in a way that correlates with whether the prompt matches the actual pain level. The model has no concept of what "mild pain" vs "severe pain" looks like — it was trained on general web videos, not clinical facial expression data.

## 9. Final Conclusion

### 9.1 What We Learned

The DaC investigation, across two experiments (vanilla + CFG-enhanced, 15 scoring combinations), provides a definitive negative result:

> **Zero-shot Diffusion-as-Classifier is fundamentally unsuitable for fine-grained facial pain classification using a general-purpose T2V model.**

This is not a tuning problem — it's a semantic gap problem. The Wan T2V model's text-visual cross-attention has no learned association between pain-level descriptions and facial micro-expressions. DaC works well for coarse-grained classification (e.g., "cat vs dog" in the original paper) because such categories have strong text-visual correspondences in web-scale training data. Pain levels do not.

### 9.2 Implications for Future Work

1. **Zero-shot text-based approaches are ruled out** for this domain. Any method relying on the pretrained model's text understanding to discriminate pain levels will fail.

2. **Supervised feature extraction (LoRA) remains the right direction.** The model needs to learn domain-specific representations from labeled BioVid data. The LoRA + classifier head pipeline (QWK=0.430) demonstrates that the model's visual features can be adapted for this task through fine-tuning.

3. **The spatial bottleneck (Section 1.1) remains the key challenge.** DaC was proposed to bypass the mean-pooling bottleneck, but its failure means we must find other solutions within the supervised paradigm: attention pooling, spatial weighting, multi-scale feature extraction, or higher input resolution.

4. **DaC scores are not useful as auxiliary features.** Given QWK≈0 and the random-chance accuracy, DaC outputs carry no discriminative signal worth combining with LoRA features.

### 9.3 Summary Table

| Method | Accuracy | QWK | Status |
|--------|----------|-----|--------|
| Random baseline | 0.200 | 0.000 | — |
| DaC vanilla (w=1) | 0.174 | -0.005 | Failed: BL1 collapse |
| DaC-CFG best (w=3) | 0.221 | +0.011 | Failed: PA3 collapse, ≈ random |
| DaC-CFG (w=5) | 0.224 | -0.006 | Failed: PA1/PA3 collapse |
| LoRA classifier (best) | 0.323 | 0.430 | Supervised baseline |

## 10. SLURM Configuration

```bash
#SBATCH --partition=gpucompute-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00

PYTHONUNBUFFERED=1   # Real-time log output
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 11. Reproducibility

### Vanilla DaC (Job 29643954)

```bash
python scripts/diffusion_classifier.py \
    --checkpoint_dir Wan2.1-T2V-14B \
    --config config_pain/config_lora_t100_aug3.yaml \
    --output_dir results/dac \
    --timesteps 200 500 800 \
    --n_noise_samples 1 \
    --split test
```

### CFG-Enhanced DaC (Job 29643988)

```bash
python scripts/diffusion_classifier_cfg.py \
    --checkpoint_dir Wan2.1-T2V-14B \
    --config config_pain/config_lora_t100_aug3.yaml \
    --output_dir results/dac_cfg \
    --timesteps 200 500 800 \
    --guidance_scales 1.0 3.0 5.0 7.5 10.0 \
    --scoring cfg_mse relative score_norm \
    --n_noise_samples 1 \
    --split test
```

Scripts: `scripts/diffusion_classifier.py` (vanilla), `scripts/diffusion_classifier_cfg.py` (CFG-enhanced). Both are self-contained, requiring only the pre-trained Wan2.1-T2V-14B checkpoint directory and a data config YAML.
