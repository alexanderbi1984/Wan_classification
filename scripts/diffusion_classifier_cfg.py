"""
CFG-Enhanced Diffusion-as-Classifier (DaC-CFG).

Key improvement over vanilla DaC: uses Classifier-Free Guidance to amplify
the text conditioning signal. In vanilla DaC, text prompts barely influence
the reconstruction (only ~7% discrimination). CFG scales up the prompt's
effect, making the model more "creative" and prompt-responsive.

Three scoring strategies:
  1. cfg_mse: MSE(x_0_guided, x_0) — reconstruction error with guided denoising
  2. relative: MSE(x_0_cond, x_0) - MSE(x_0_uncond, x_0) — marginal improvement
  3. score_norm: ||v_cond - v_uncond||² — how strongly each prompt "pulls"

Flow matching with CFG:
    v_uncond = DiT(x_t, t, ∅)
    v_cond   = DiT(x_t, t, prompt_k)
    v_guided = v_uncond + w * (v_cond - v_uncond)       # w = guidance scale
    x_0_pred = x_t - sigma * v_guided

Usage:
    python scripts/diffusion_classifier_cfg.py \
        --checkpoint_dir Wan2.1-T2V-14B \
        --config config_pain/config_lora_t100_aug3.yaml \
        --output_dir results/dac_cfg \
        --guidance_scales 1.0 3.0 5.0 7.5 \
        --timesteps 200 500 800 \
        --scoring cfg_mse relative score_norm
"""

import argparse
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.online_video import read_frames_from_directory
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

CLASS_NAMES = ["BL1", "PA1", "PA2", "PA3", "PA4"]

CLASS_PROMPTS = {
    0: [
        "A video of a person's face with a neutral expression, showing no pain.",
        "A calm face with no signs of discomfort or pain response.",
        "Baseline recording of a relaxed person with no painful stimulation.",
    ],
    1: [
        "A video showing a face with very mild pain, slight discomfort.",
        "A person experiencing low-intensity pain with minimal facial reaction.",
        "Subtle facial response to mild pain stimulation.",
    ],
    2: [
        "A video showing moderate pain expression on a person's face.",
        "Visible grimacing from medium-intensity pain stimulation.",
        "Clear facial discomfort from moderate pain.",
    ],
    3: [
        "A video showing strong pain expression with pronounced grimacing.",
        "Intense facial contortion from high pain stimulation.",
        "Significant facial distress from severe pain.",
    ],
    4: [
        "A video showing extreme pain with severe facial contortion.",
        "Agonized facial expression from very high pain stimulation.",
        "Maximum pain response with intense grimacing and facial distress.",
    ],
}


def add_noise(x_0, noise, sigma):
    return (1 - sigma) * x_0 + sigma * noise


def load_models(checkpoint_dir, device="cuda"):
    print(f"[DaC-CFG] Loading models from {checkpoint_dir}")
    vae = WanVAE(
        vae_pth=os.path.join(checkpoint_dir, "Wan2.1_VAE.pth"),
        device=device,
    )
    t5_encoder = T5EncoderModel(
        text_len=512, dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(checkpoint_dir, "google", "umt5-xxl"),
    )
    t5_encoder.model.to(device)

    dit = WanModel.from_pretrained(checkpoint_dir)
    dit.eval().requires_grad_(False).to(device)

    alloc = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[DaC-CFG] Models loaded. VRAM: {alloc:.1f} / {total:.1f} GiB")
    return dit, vae, t5_encoder


def precompute_embeddings(t5_encoder, device):
    """Pre-compute class embeddings + null (unconditional) embedding."""
    print("[DaC-CFG] Pre-computing embeddings...")
    class_embeddings = {}
    for cls, prompts in CLASS_PROMPTS.items():
        embs = t5_encoder(prompts, device)
        max_len = max(e.shape[0] for e in embs)
        padded = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embs])
        class_embeddings[cls] = padded.mean(dim=0)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {class_embeddings[cls].shape}")

    null_emb = t5_encoder([""], device)
    class_embeddings["null"] = null_emb[0]
    print(f"  Null (unconditional): {class_embeddings['null'].shape}")

    return class_embeddings


def encode_video(vae, frames_root, video_id, resize=128, max_frames=129,
                 sample_rate=2, device="cuda"):
    frame_dir = os.path.join(frames_root, video_id)
    frames = read_frames_from_directory(
        frame_dir, max_frames=max_frames, resize=resize, sample_rate=sample_rate
    )
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
    with torch.no_grad():
        latents = vae.encode([video_tensor.to(device)])
    return latents[0]


def classify_single_cfg(dit, latent, class_embeddings, sigma_values, t_values,
                        guidance_scales, scoring_methods, n_noise_samples=1,
                        device="cuda"):
    """Classify with CFG-enhanced DaC.

    Returns:
        results: dict keyed by (guidance_scale, scoring_method) → {pred, losses}
    """
    n_classes = 5
    _, T_lat, H_lat, W_lat = latent.shape
    seq_len = T_lat * (H_lat // 2) * (W_lat // 2)

    null_ctx = class_embeddings["null"].to(device=device, dtype=latent.dtype)

    # Accumulators: per (guidance_scale, scoring_method, class)
    accum = {}
    for w in guidance_scales:
        for method in scoring_methods:
            accum[(w, method)] = {cls: 0.0 for cls in range(n_classes)}
    n_evals = 0

    for sigma, t_val in zip(sigma_values, t_values):
        for _ in range(n_noise_samples):
            noise = torch.randn_like(latent)
            x_t = add_noise(latent, noise, sigma)

            # Unconditional forward pass (batch size 1)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_uncond = dit([x_t.clone()],
                               torch.full((1,), t_val, device=device, dtype=latent.dtype),
                               [null_ctx], seq_len)[0]

            # Conditional forward pass for all 5 classes (batched)
            x_list = [x_t.clone() for _ in range(n_classes)]
            t_tensor = torch.full((n_classes,), t_val, device=device, dtype=latent.dtype)
            ctx_list = [
                class_embeddings[cls].to(device=device, dtype=latent.dtype)
                for cls in range(n_classes)
            ]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_conds = dit(x_list, t_tensor, ctx_list, seq_len)

            # Compute all scoring variants
            x0_uncond = x_t - sigma * v_uncond
            mse_uncond = F.mse_loss(x0_uncond, latent).item()

            for cls in range(n_classes):
                v_cond = v_conds[cls]
                x0_cond = x_t - sigma * v_cond
                mse_cond = F.mse_loss(x0_cond, latent).item()

                # Score difference: ||v_cond - v_uncond||²
                score_diff_norm = F.mse_loss(v_cond, v_uncond).item()

                for w in guidance_scales:
                    # CFG-guided reconstruction
                    v_guided = v_uncond + w * (v_cond - v_uncond)
                    x0_guided = x_t - sigma * v_guided
                    mse_guided = F.mse_loss(x0_guided, latent).item()

                    for method in scoring_methods:
                        if method == "cfg_mse":
                            accum[(w, method)][cls] += mse_guided
                        elif method == "relative":
                            accum[(w, method)][cls] += (mse_cond - mse_uncond)
                        elif method == "score_norm":
                            # Higher norm = prompt has stronger effect
                            # We want argMAX (strongest pull), so negate for argmin
                            accum[(w, method)][cls] += (-score_diff_norm)

            n_evals += 1

    results = {}
    for (w, method), cls_losses in accum.items():
        avg = {cls: loss / n_evals for cls, loss in cls_losses.items()}
        pred = min(avg, key=avg.get)
        results[(w, method)] = {"pred": pred, "losses": avg}

    return results


def run_evaluation(dit, vae, class_embeddings, test_df, data_cfg,
                   sigma_values, t_values, guidance_scales, scoring_methods,
                   n_noise_samples, device):
    """Run full evaluation, collecting results for all (w, method) combos."""
    combos = [(w, m) for w in guidance_scales for m in scoring_methods]
    all_results = {combo: {"preds": [], "labels": [], "losses": []} for combo in combos}

    start_time = time.time()
    n_total = len(test_df)

    for idx, row in test_df.iterrows():
        video_id = row["video_id"]
        true_label = int(row["pain_level"])

        try:
            latent = encode_video(
                vae, data_cfg["frames_root"], video_id,
                resize=data_cfg.get("resize", 128),
                max_frames=data_cfg.get("max_frames", 129),
                sample_rate=data_cfg.get("sample_rate", 2),
                device=device,
            )

            sample_results = classify_single_cfg(
                dit, latent, class_embeddings,
                sigma_values, t_values,
                guidance_scales, scoring_methods,
                n_noise_samples=n_noise_samples,
                device=device,
            )

            for combo in combos:
                r = sample_results[combo]
                all_results[combo]["preds"].append(r["pred"])
                all_results[combo]["labels"].append(true_label)
                all_results[combo]["losses"].append(r["losses"])

            n_done = len(all_results[combos[0]]["preds"])
            elapsed = time.time() - start_time
            speed = n_done / elapsed
            eta = (n_total - n_done) / max(speed, 0.01)

            if n_done % 20 == 0 or n_done == 1:
                # Show running accuracy for each combo
                summary_parts = []
                for combo in combos:
                    acc = accuracy_score(
                        all_results[combo]["labels"],
                        all_results[combo]["preds"]
                    )
                    summary_parts.append(f"w={combo[0]}/{combo[1]}:{acc:.3f}")
                summary = " | ".join(summary_parts)
                print(f"  [{n_done}/{n_total}] {video_id} "
                      f"true={CLASS_NAMES[true_label]} | {speed:.1f} s/s | "
                      f"ETA: {eta/60:.0f}min")
                print(f"    {summary}")

        except Exception as e:
            print(f"  ERROR {video_id}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def compute_metrics(labels, preds):
    y_true = np.array(labels)
    y_pred = np.array(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="CFG-Enhanced DaC classifier")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/dac_cfg")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[200, 500, 800])
    parser.add_argument("--guidance_scales", type=float, nargs="+",
                        default=[1.0, 3.0, 5.0, 7.5],
                        help="CFG guidance scales to test (1.0 = no guidance = vanilla DaC)")
    parser.add_argument("--scoring", type=str, nargs="+",
                        default=["cfg_mse", "relative", "score_norm"],
                        choices=["cfg_mse", "relative", "score_norm"])
    parser.add_argument("--n_noise_samples", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    os.makedirs(args.output_dir, exist_ok=True)

    dit, vae, t5_encoder = load_models(args.checkpoint_dir, args.device)
    class_embeddings = precompute_embeddings(t5_encoder, args.device)

    t5_encoder.model.cpu()
    del t5_encoder
    torch.cuda.empty_cache()
    print("[DaC-CFG] T5 unloaded.")

    sigma_values = [t / 1000.0 for t in args.timesteps]
    for t, s in zip(args.timesteps, sigma_values):
        print(f"  t={t:.0f} → σ={s:.3f}")

    df = pd.read_csv(data_cfg["labels_csv"])
    test_df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples)

    n_combos = len(args.guidance_scales) * len(args.scoring)
    print(f"\n[DaC-CFG] Evaluating {len(test_df)} samples")
    print(f"  Guidance scales: {args.guidance_scales}")
    print(f"  Scoring methods: {args.scoring}")
    print(f"  Total combos: {n_combos}")
    print(f"  Forward passes per sample: {len(args.timesteps) * args.n_noise_samples} "
          f"× (1 uncond + 5 cond) = {len(args.timesteps) * args.n_noise_samples * 6}")

    all_results = run_evaluation(
        dit, vae, class_embeddings, test_df, data_cfg,
        sigma_values, [t for t in args.timesteps],
        args.guidance_scales, args.scoring,
        args.n_noise_samples, args.device,
    )

    # Print results summary
    print(f"\n{'='*80}")
    print(f"[DaC-CFG] Results Summary")
    print(f"{'='*80}")

    summary_rows = []
    for (w, method), data in sorted(all_results.items()):
        metrics = compute_metrics(data["labels"], data["preds"])
        summary_rows.append({
            "guidance_scale": w, "scoring": method, **metrics
        })
        print(f"\n  w={w:.1f} | {method}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}  "
              f"Macro-F1: {metrics['macro_f1']:.4f}  "
              f"QWK: {metrics['qwk']:.4f}  "
              f"MAE: {metrics['mae']:.4f}")

        y_true = np.array(data["labels"])
        y_pred = np.array(data["preds"])
        pred_dist = np.bincount(y_pred, minlength=5)
        print(f"    Pred dist: {dict(zip(CLASS_NAMES, pred_dist.tolist()))}")

    # Find best combo
    best = max(summary_rows, key=lambda r: r["qwk"])
    print(f"\n{'='*80}")
    print(f"  BEST: w={best['guidance_scale']:.1f} / {best['scoring']} "
          f"→ Acc={best['accuracy']:.4f} QWK={best['qwk']:.4f}")
    print(f"{'='*80}")

    # Save full results
    output = {
        "method": "DaC-CFG",
        "checkpoint_dir": args.checkpoint_dir,
        "timesteps": args.timesteps,
        "n_noise_samples": args.n_noise_samples,
        "guidance_scales": args.guidance_scales,
        "scoring_methods": args.scoring,
        "split": args.split,
        "n_samples": len(test_df),
        "summary": summary_rows,
        "best": best,
        "per_combo": {},
    }

    for (w, method), data in all_results.items():
        key = f"w{w}_{method}"
        metrics = compute_metrics(data["labels"], data["preds"])
        y_true = np.array(data["labels"])
        y_pred = np.array(data["preds"])
        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

        output["per_combo"][key] = {
            "guidance_scale": w,
            "scoring": method,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "predictions": [
                {
                    "video_id": test_df.iloc[i]["video_id"],
                    "true": int(data["labels"][i]),
                    "pred": int(data["preds"][i]),
                    "losses": data["losses"][i],
                }
                for i in range(len(data["preds"]))
            ],
        }

    results_path = os.path.join(args.output_dir, "dac_cfg_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DaC-CFG] Results saved to {results_path}")


if __name__ == "__main__":
    main()
