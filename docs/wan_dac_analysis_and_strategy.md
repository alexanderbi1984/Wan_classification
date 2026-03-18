# Wan Video Diffusion for Pain Classification: Comprehensive Analysis & Strategic Direction

**Date**: March 17, 2026  
**Status**: Active analysis

---

## 1. Executive Summary

After 22+ experiments across two distinct approaches — **WanLoRA Feature Extraction** and **Diffusion-as-Classifier (DaC)** — we have established that Wan's utility for fine-grained 5-class pain classification (BL1/PA1/PA2/PA3/PA4) is fundamentally limited. Both approaches hit distinct but related ceilings. This document synthesizes all findings and proposes a focused strategy: **binary classification (BL1 vs PA4)** to demonstrate Wan's temporal modeling strength on a tractable problem.

---

## 2. Approach 1: WanLoRA Feature Extraction — What Worked and What Didn't

### 2.1 Best Results (22 experiments)

| Metric | Best Value | Experiment |
|--------|-----------|-----------|
| Test QWK | **0.407** | t100_aug3 (CORAL, dim=128, t=100, aug3) |
| Test Accuracy | **0.302** | dim256_aug2 |
| BL1 Recall | **0.211** | pure CE (but QWK drops to 0.334) |

### 2.2 The BL1 Problem — Root Cause Confirmed

Binary classification diagnostics revealed a **monotonic separability gradient**:

```
BL1 vs PA4: QWK=0.468, BL1 recall=83.9%  ← strong separation
BL1 vs PA3: QWK=0.251, BL1 recall=85.0%  ← moderate
BL1 vs PA2: QWK=-0.03, BL1 recall=17.2%  ← near-random
BL1 vs PA1: QWK=0.000, BL1 recall=0.0%   ← complete failure (loss stuck at 0.693)
```

The Wan DiT backbone **cannot distinguish BL1 from PA1** in its latent space — even with LoRA fine-tuning, the extracted features for these two classes are identical.

### 2.3 Grad-CAM Analysis — Spatial Information Bottleneck

Grad-CAM visualizations confirmed the architectural root cause:

| Pain Level | Model Attention | Feature Quality |
|-----------|----------------|----------------|
| **PA4** (severe) | Eyes, nose, mouth — core face | Genuine pain features |
| **PA3** (high) | Face (emerges in later frames) | Temporal dynamics captured |
| **PA2** (medium) | Weak, inconsistent | Unstable |
| **PA1** (low) | **Corners and image edges** | Essentially random |
| **BL1** (none) | **Random / corners** | No signal |

The compression pipeline is the bottleneck:

```
128×128 face → VAE(8×) → 16×16 → DiT(patch 2×) → 8×8 → mean(H,W) → 1×1
```

Pain-relevant Action Units (AU4, AU6, AU7, AU43) occupy 5–15 pixels in the input. After compression to a 4×4 spatial grid (each cell covering 32×32 pixels), these micro-expression signals fall **below the resolution floor**.

**However**: For PA3/PA4, the model successfully learns temporal dynamics of pain onset — attention builds over time across frames, demonstrating genuine temporal understanding.

### 2.4 Performance Ceiling Established

After trying every reasonable variation (loss functions, augmentation levels, LoRA rank, attention pooling, multi-layer features, TTA, ensemble), the best Test QWK remains at 0.407. The ceiling is set by the backbone's feature representation, not by training methodology.

---

## 3. Approach 2: Diffusion-as-Classifier (DaC) — Why It Failed

### 3.1 Results

| Metric | DaC (zero-shot) | LoRA Classifier (best) | Random Baseline |
|--------|-----------------|----------------------|-----------------|
| Accuracy | 0.174 | 0.302 | 0.200 |
| QWK | **-0.005** | 0.407 | 0.000 |

**Prediction collapse**: 71% of all predictions are BL1.

### 3.2 Failure Analysis: Three Fundamental Problems

#### Problem 1: Text Conditioning Is Too Weak

Five class-specific text prompts produce nearly identical reconstruction errors — only ~7% discrimination:

```
BL1 prompt avg loss: 0.02528  ← lowest (always wins)
PA1 prompt avg loss: 0.02627
PA3 prompt avg loss: 0.02595
PA4 prompt avg loss: 0.02656
PA2 prompt avg loss: 0.02664  ← highest
```

At low noise levels (σ=0.2), `x_t = 0.8·x_0 + 0.2·noise` — the model can reconstruct from the visual signal alone without "listening" to the text prompt at all.

#### Problem 2: Domain Mismatch — Wan Has No Clinical Pain Priors

DaC's success on ImageNet (Li et al., ICLR 2024) relies on the diffusion model having strong generative priors for each class. Wan was trained on internet-scale video data and likely learned:

- ✅ Basic facial expressions (happy, sad, angry, surprised)
- ✅ Extreme pain (screaming, contortion — from movies/drama)
- ❌ **Clinical pain levels** (the difference between BL1's neutral face and PA1's slightly furrowed brow)

The T5 encoder understands "severe pain" semantically, but the DiT's text-to-visual mapping for fine-grained pain expressions was **never trained**. Even with CFG amplification (guidance scale up to 10.0), this domain gap cannot be bridged at inference time.

#### Problem 3: Intra-Class Variance — Pain Expression Is Highly Individual

DaC implicitly assumes each class has a single "template." In reality:

```
Patient A (stoic)      + PA4 stimulation → subtle brow furrow
Patient B (expressive) + PA4 stimulation → face contortion

Wan's "severe pain" template (from movie data):
  → expressive contortion (biased toward dramatic expression)

DaC judgment:
  Patient A + "severe pain" prompt → hallucinated contortion ≠ subtle furrow → high MSE → ❌
  Patient B + "severe pain" prompt → hallucinated contortion ≈ actual contortion → low MSE → ✅
```

This means DaC would systematically misclassify stoic pain expressors, who are common in clinical settings (especially elderly patients and males).

### 3.3 CFG Enhancement — Potential but Limited

A CFG-enhanced DaC variant (`scripts/diffusion_classifier_cfg.py`) was developed with three scoring strategies:

1. **cfg_mse**: MSE with guidance-amplified reconstruction
2. **relative**: Conditional improvement over unconditional baseline  
3. **score_norm**: Prompt "pull" strength measurement

While CFG can amplify text conditioning signal from 7% to potentially 30-50%, it cannot create priors that don't exist. If Wan doesn't know what PA2 looks like, no amount of guidance scale will fix it.

---

## 4. Synthesized Root Cause

Both approaches fail for the same fundamental reason:

> **Wan is a video generation model, not a face understanding model.**

Its features are optimized for **generating visually plausible video from text**, not for **discriminating subtle facial action unit changes**. This manifests as:

- **WanLoRA**: Spatial compression destroys micro-expression signals; attention drifts to image edges for low-pain classes
- **DaC**: Text-to-visual mapping lacks clinical pain granularity; reconstruction error reflects facial geometric complexity, not content match

**However**, Wan does demonstrate genuine strength in one area: **temporal dynamics**. The Grad-CAM analysis shows that for PA3/PA4, the model correctly learns that pain builds over time — attention maps evolve from minimal activation in early frames to strong face-centered activation in later frames. This temporal understanding is the model's distinguishing capability.

---

## 5. Strategic Direction: Binary Classification (BL1 vs PA4)

### 5.1 Rationale — Play to Wan's Strengths

Given the analysis above, the optimal strategy is **扬长避短** (leverage strengths, avoid weaknesses):

| Aspect | 5-Class Problem | BL1 vs PA4 Binary |
|--------|----------------|-------------------|
| Spatial resolution needed | 5-15px micro-expressions | Whole-face expressions (50px+) |
| Existing separability | QWK=0.407 (near ceiling) | **QWK=0.468** (with room to improve) |
| Grad-CAM evidence | Attention on face for PA4 ✅ | Attention on face for PA4 ✅ |
| Temporal signal | Strong for PA4 onset | Strong for PA4 onset ✅ |
| Clinical relevance | Fine-grained triage | Pain detection (binary screening) |

Binary BL1 vs PA4 is the problem where:
1. **Wan's temporal modeling is most valuable** — PA4 has clear temporal pain onset patterns
2. **Spatial bottleneck is least harmful** — PA4 expressions are large enough to survive compression
3. **Has highest existing separability** — already QWK=0.468 in pilot binary experiment
4. **Is clinically meaningful** — binary pain detection is a real-world screening application

### 5.2 Why This Can Beat SOTA

BioVid binary (BL1 vs PA4) SOTA results from the literature typically use:
- Frame-level features (no temporal modeling)
- Face-specialized but temporally naive architectures
- Small backbone models (ResNet, VGG)

Wan brings **14 billion parameters pre-trained on video temporal dynamics** — a fundamentally different (and potentially superior) source of temporal understanding. If we can demonstrate state-of-the-art on BL1 vs PA4, it validates the hypothesis that **video diffusion models capture temporal pain dynamics that frame-based methods miss**.

### 5.3 Experiment Plan

Use the established WanLoRA pipeline (best config: CORAL, dim=128, t=100, aug3) with:
- Fixed train/val/test split (not LOSO, for initial validation)
- Binary cross-entropy or CORAL-binary loss
- Same augmentation strategy (aug3)
- Compare against our existing binary pilot result (QWK=0.468)

### 5.4 What This Means for the Paper

If WanLoRA achieves strong BL1 vs PA4 performance:
- **Positive result**: Video diffusion backbones provide superior temporal features for pain onset detection
- **The 5-class analysis becomes a valuable negative result**: explains *why* fine-grained classification fails (spatial bottleneck + domain mismatch) and establishes the boundary of applicability
- **DaC analysis provides additional insight**: demonstrates that zero-shot classification with generation models requires domain-specific priors

---

## 6. Future Directions Beyond Binary

If binary BL1 vs PA4 succeeds:

1. **Progressive difficulty**: BL1 vs PA3, BL1 vs PA2, to map Wan's discrimination boundary
2. **3-class formulation**: BL1 vs PA1+PA2 (merged) vs PA3+PA4 (merged)
3. **Hybrid architecture**: Wan temporal features + face-specialized spatial features (ArcFace/InsightFace)
4. **Higher resolution**: 256×256 input (32×32 latent, 16×16 DiT features) to push the spatial bottleneck
5. **LOSO cross-validation**: Once the best config is identified on fixed split

---

## Appendix: Key Files Reference

| File | Description |
|------|------------|
| `scripts/diffusion_classifier.py` | Vanilla DaC implementation |
| `scripts/diffusion_classifier_cfg.py` | CFG-enhanced DaC with 3 scoring strategies |
| `docs/experiment_results.md` | Full 22-experiment results table |
| `docs/gradcam_analysis_report.md` | Grad-CAM attention analysis |
| `docs/diffusion_as_classifier_report.md` | DaC technical report and failure analysis |
