# Grad-CAM Feature Attribution Analysis Report

**Date**: March 17, 2026  
**Status**: Completed

---

## 1. Motivation

After extensive experiments showing consistently poor BL1 (no pain) recall and a monotonic separability gradient across pain levels, we needed a visual explanation of **where the model focuses** when making predictions. Grad-CAM (Gradient-weighted Class Activation Mapping) provides spatial heatmaps that reveal which image regions contribute most to the model's classification decisions.

The key question: **Does the Wan DiT backbone actually learn to look at facial features, or is it relying on spurious correlations?**

## 2. Implementation

### 2.1 Script

`scripts/visualize_gradcam.py` — Generates spatial heatmaps by hooking into the DiT feature extraction stage.

### 2.2 Architecture Flow for Grad-CAM

```
video (B,C,T,H,W) → WanVAE.encode() → vae_latents (B,16,T',H',W')
                                       → WanDiT [LoRA] → dit_features (B,dim,T',H',W')  ← HOOK HERE
                                       → XDiTProcessor → TemporalEncoder → Pool → Head → logits
                                       
Gradient:  target_logit → backprop → grad w.r.t. dit_features
Grad-CAM:  channel_weights = GAP(grad);  cam = ReLU(Σ(w × feat))
```

### 2.3 Key Implementation Details

| Aspect | Detail |
|--------|--------|
| Hook location | DiT output features `(B, 5120, T', H', W')` |
| Gradient target | CORAL cumulative probability for predicted class |
| Channel weighting | Global Average Pooling of gradients over spatial+temporal dims |
| Spatial resolution | 4×4 grid (after VAE 8× + DiT patch 2× compression) |
| Upsampling | Bilinear interpolation from 4×4 → 128×128 |
| Model precision | float32 (requires H100 96GB; bfloat16 caused dtype mismatches) |
| GRU workaround | `model.temporal_encoder.train()` to enable cuDNN backward in eval mode |
| Visualization | 3 rows (original, heatmap, overlay) × 9 columns (8 frames + temporal average) |
| Organization | Results separated into `correct/` and `wrong/` subdirectories |

### 2.4 Checkpoint Used

Best model from fixed-split experiments:
```
results/wan_lora_14B_biovid_t100_aug3_20260305_002030/checkpoints/
  wan_lora_14B_biovid_t100_aug3_best-epoch=14-val_pain_QWK=0.430.ckpt
```

### 2.5 Sample Selection

- 10 samples per class × 5 classes = **50 test samples**
- Randomly sampled with seed=123 for reproducibility
- Results: **14 correct (28%) / 36 wrong (72%)**

## 3. Critical Findings

### Finding 1: Pain Intensity Determines Feature Relevance

The most important finding from Grad-CAM analysis is a clear **correlation between pain intensity and the model's ability to focus on relevant facial regions**:

| Pain Level | Heatmap Location | Model Behavior | Confidence |
|-----------|-----------------|----------------|------------|
| **PA4** (severe) | Eyes, nose, mouth — core face | Learns genuine pain features | c4=0.60–0.77, highly confident |
| **PA3** (high) | Face (emerges in later frames) | Partially learns, time-dependent | c3=0.30–0.39, moderate |
| **PA2** (medium) | Weak, inconsistent | Unstable feature selection | Near-uniform distribution |
| **PA1** (low) | **Corners and image edges** | Essentially guessing | c0≈c1≈0.30–0.40, near-random |
| **BL1** (none) | **Random / corners** | Completely random | Uniform across all classes |

### Finding 2: Temporal Dynamics of Attention

For correctly classified high-pain samples (PA3, PA4), the heatmaps exhibit a clear **temporal evolution**:
- **Early frames (t=0–7)**: Minimal activation — subject at baseline
- **Mid frames (t=7–12)**: Attention gradually builds on face
- **Late frames (t=12–17)**: Strong face-centered activation — pain expression peaks

This demonstrates that the model has learned the temporal dynamics of pain onset, at least for severe pain.

### Finding 3: Age/Wrinkle Confusion

BL1 sample `120514_w_56` (56-year-old female) was predicted as PA3 (high pain) with c4=0.37. The Grad-CAM shows the model attending to natural facial wrinkles and age-related features, mistaking them for pain expressions. This reveals a fundamental confound: the model cannot distinguish **structural facial features** (wrinkles, sagging) from **dynamic pain expressions** (grimacing, furrowing).

### Finding 4: Per-Subject Consistency

Within the same subject across different pain levels, the heatmap patterns are remarkably consistent. For subject `083109_m_60`, all pain levels show attention on the chin/nose area, suggesting the model is partially recognizing subject identity rather than pain expression.

## 4. Root Cause Analysis: The Spatial Compression Pipeline

The Grad-CAM analysis provides visual proof of a fundamental architectural limitation:

```
Input:     128 × 128 pixels (face image)
    ↓ VAE (stride 8×8)
Latent:    16 × 16 grid
    ↓ DiT (patch_size 2×2)
Tokens:    8 × 8 grid  →  then  4 × 4 after internal processing
    ↓ XDiTProcessor
           mean(dim=[H,W])  →  ALL spatial info discarded
```

- **128→16→8→4→1**: Five stages of spatial compression
- Each token in the 4×4 Grad-CAM grid covers a **32×32 pixel** region of the face
- Pain-relevant Action Units (AU4, AU6, AU7, AU43) occupy regions of **5–15 pixels**
- These micro-expression details are **below the resolution floor** of the feature pipeline

### Contrast with SOTA Approaches

| | Wan DiT (ours) | Face-specialized (SOTA) |
|--|---|---|
| Input resolution | 128×128 | 224×224 |
| Spatial bottleneck | 4×4 (1,024× compression) | 7×7 (1,024× compression) |
| Pre-training objective | Video generation (denoising) | Face recognition (discriminative) |
| Feature type | Generative latent space | Discriminative embedding space |
| Micro-expression capture | Lost in compression | Preserved through discriminative training |

## 5. Implications

1. **The Wan DiT backbone genuinely learns pain-relevant features, but only for severe pain.** This validates the approach for high-pain detection (PA3/PA4) but explains the failure for subtle pain (BL1/PA1).

2. **Spatial information loss is the primary bottleneck**, not model capacity. The 14B parameters are more than sufficient; the information is simply not present in the compressed representation.

3. **Future work** should focus on preserving spatial resolution:
   - Higher input resolution (256×256 → 32×32 latent)
   - VAE intermediate layer features (access pre-compression representations)
   - Hybrid architecture (face-specialized backbone for spatial + DiT for temporal)
   - Diffusion-as-Classifier (uses full-resolution reconstruction error instead of compressed features)

## 6. Output

All 50 visualizations are saved in `results/gradcam/`:
- `results/gradcam/correct/` — 14 correctly classified samples
- `results/gradcam/wrong/` — 36 incorrectly classified samples

Each PNG contains: 3 rows (original frame, heatmap, overlay) × 9 columns (8 temporal frames + temporal average), with title showing true/predicted class and per-class probabilities.
