"""
Diffusion-as-Classifier (DaC): Zero-shot pain classification using Wan DiT.

Instead of extracting features and training a classifier head, this approach
uses the diffusion model's denoising capability directly for classification.
For each video, noise is added at multiple timesteps, and the DiT is asked to
denoise conditioned on different pain-level text prompts. The prompt that
yields the lowest reconstruction error = the predicted class.

Based on: "Your Diffusion Model is Secretly a Zero-Shot Classifier" (Li et al.)

Flow matching formulation:
    x_t = (1 - sigma) * x_0 + sigma * noise
    model predicts flow v such that: x_0_pred = x_t - sigma * v
    loss = ||x_0_pred - x_0||^2

Usage:
    python scripts/diffusion_classifier.py \
        --checkpoint_dir Wan2.1-T2V-14B \
        --config config_pain/config_lora_t100_aug3.yaml \
        --output_dir results/dac \
        --timesteps 200 500 800 \
        --n_noise_samples 3
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
from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

CLASS_NAMES = ["BL1", "PA1", "PA2", "PA3", "PA4"]

# Pain-level prompt pools — semantically varied descriptions per class
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


def build_noise_schedule(num_train_timesteps=1000, shift=5.0):
    """Build the flow matching noise schedule with timestep shifting."""
    alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
    sigmas = 1.0 - alphas
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = torch.from_numpy(sigmas).float()
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps


def add_noise(x_0, noise, sigma):
    """Add noise using flow matching: x_t = (1 - sigma) * x_0 + sigma * noise."""
    return (1 - sigma) * x_0 + sigma * noise


def load_models(checkpoint_dir, device="cuda"):
    """Load Wan DiT, VAE, and T5 encoder."""
    print(f"[DaC] Loading models from {checkpoint_dir}")

    # VAE
    vae_path = os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
    print(f"[DaC] Loading VAE: {vae_path}")
    vae = WanVAE(vae_pth=vae_path, device=device)

    # T5 encoder
    t5_ckpt = os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tok = os.path.join(checkpoint_dir, "google", "umt5-xxl")
    print(f"[DaC] Loading T5 encoder: {t5_ckpt}")
    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=t5_ckpt,
        tokenizer_path=t5_tok,
    )
    t5_encoder.model.to(device)

    # DiT model
    print(f"[DaC] Loading WanModel (DiT) from: {checkpoint_dir}")
    dit_model = WanModel.from_pretrained(checkpoint_dir)
    dit_model.eval().requires_grad_(False)
    dit_model.to(device)

    alloc = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[DaC] Models loaded. VRAM: {alloc:.1f} / {total:.1f} GiB")

    return dit_model, vae, t5_encoder


def precompute_class_embeddings(t5_encoder, device, strategy="mean"):
    """Pre-compute T5 embeddings for each pain class.

    Args:
        strategy: 'mean' averages all prompts per class into one embedding,
                  'all' keeps all prompts (will be averaged during classification).
    """
    print("[DaC] Pre-computing T5 embeddings for class prompts...")
    class_embeddings = {}

    for cls, prompts in CLASS_PROMPTS.items():
        embeddings = t5_encoder(prompts, device)  # list of [L_i, 4096]
        if strategy == "mean":
            max_len = max(e.shape[0] for e in embeddings)
            padded = torch.stack([
                F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embeddings
            ])
            class_embeddings[cls] = padded.mean(dim=0)  # [max_len, 4096]
        else:
            class_embeddings[cls] = embeddings
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): embedding shape {class_embeddings[cls].shape}")

    return class_embeddings


def encode_video(vae, frames_root, video_id, resize=128, max_frames=129,
                 sample_rate=2, device="cuda"):
    """Encode a single video to VAE latent space."""
    frame_dir = os.path.join(frames_root, video_id)
    frames = read_frames_from_directory(
        frame_dir, max_frames=max_frames, resize=resize, sample_rate=sample_rate
    )
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

    with torch.no_grad():
        latents = vae.encode([video_tensor.to(device)])
    return latents[0]  # (16, T', H', W') float32


def classify_single(dit_model, latent, class_embeddings, sigma_values,
                    t_values, n_noise_samples=1, device="cuda"):
    """Classify a single video using Diffusion-as-Classifier.

    Returns:
        pred_class: predicted class index
        class_losses: dict of {class_idx: avg_reconstruction_loss}
    """
    n_classes = len(class_embeddings)
    accumulated_losses = {cls: 0.0 for cls in range(n_classes)}
    n_evals = 0

    # Compute seq_len from latent shape
    _, T_lat, H_lat, W_lat = latent.shape
    # After patch embedding (stride 1,2,2): tokens = T_lat * (H_lat//2) * (W_lat//2)
    seq_len = T_lat * (H_lat // 2) * (W_lat // 2)

    for sigma, t_val in zip(sigma_values, t_values):
        for _ in range(n_noise_samples):
            noise = torch.randn_like(latent)
            x_t = add_noise(latent, noise, sigma)

            # Batch all 5 classes together: replicate x_t and pair with each class embedding
            x_list = [x_t.clone() for _ in range(n_classes)]
            t_tensor = torch.full((n_classes,), t_val, device=device, dtype=latent.dtype)
            context_list = [
                class_embeddings[cls].to(device=device, dtype=latent.dtype)
                for cls in range(n_classes)
            ]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = dit_model(x_list, t_tensor, context_list, seq_len)

            for cls in range(n_classes):
                flow_pred = outputs[cls]  # (16, T', H', W')
                x0_pred = x_t - sigma * flow_pred
                recon_loss = F.mse_loss(x0_pred, latent).item()
                accumulated_losses[cls] += recon_loss

            n_evals += 1

    avg_losses = {cls: loss / n_evals for cls, loss in accumulated_losses.items()}
    pred_class = min(avg_losses, key=avg_losses.get)
    return pred_class, avg_losses


def main():
    parser = argparse.ArgumentParser(description="Diffusion-as-Classifier for pain recognition")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to Wan2.1-T2V-14B checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Data config YAML (for frames_root, labels_csv, etc.)")
    parser.add_argument("--output_dir", type=str, default="results/dac")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[200, 500, 800],
                        help="Timestep values for noise injection (t = sigma * 1000)")
    parser.add_argument("--n_noise_samples", type=int, default=3,
                        help="Number of random noise samples per timestep")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to first N samples (for quick testing)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_strategy", type=str, default="mean",
                        choices=["mean"], help="How to combine multiple prompts per class")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    dit_model, vae, t5_encoder = load_models(args.checkpoint_dir, device=args.device)

    # Pre-compute text embeddings
    class_embeddings = precompute_class_embeddings(
        t5_encoder, device=args.device, strategy=args.prompt_strategy
    )

    # Free T5 encoder VRAM after pre-computation
    t5_encoder.model.cpu()
    del t5_encoder
    torch.cuda.empty_cache()
    print("[DaC] T5 encoder unloaded to free VRAM.")

    # Build noise schedule
    sigmas, timesteps_schedule = build_noise_schedule()

    # Convert user-specified timestep values to sigma
    sigma_values = []
    t_values = []
    for t_val in args.timesteps:
        sigma = t_val / 1000.0
        sigma_values.append(sigma)
        t_values.append(t_val)
        print(f"  Timestep t={t_val:.0f} → sigma={sigma:.3f} (noise level: {sigma*100:.1f}%)")

    # Load dataset
    df = pd.read_csv(data_cfg["labels_csv"])
    test_df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples)

    print(f"\n[DaC] Evaluating {len(test_df)} {args.split} samples")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Noise samples per timestep: {args.n_noise_samples}")
    print(f"  Total forward passes per sample: {len(args.timesteps) * args.n_noise_samples} "
          f"(batched across {len(CLASS_NAMES)} classes)")

    # Evaluation loop
    all_preds = []
    all_labels = []
    all_losses = []
    start_time = time.time()

    for idx, row in test_df.iterrows():
        video_id = row["video_id"]
        true_label = int(row["pain_level"])

        try:
            latent = encode_video(
                vae, data_cfg["frames_root"], video_id,
                resize=data_cfg.get("resize", 128),
                max_frames=data_cfg.get("max_frames", 129),
                sample_rate=data_cfg.get("sample_rate", 2),
                device=args.device,
            )

            pred_class, class_losses = classify_single(
                dit_model, latent, class_embeddings,
                sigma_values, t_values,
                n_noise_samples=args.n_noise_samples,
                device=args.device,
            )

            all_preds.append(pred_class)
            all_labels.append(true_label)
            all_losses.append(class_losses)

            elapsed = time.time() - start_time
            speed = (len(all_preds)) / elapsed
            eta = (len(test_df) - len(all_preds)) / max(speed, 0.01)

            if (len(all_preds)) % 20 == 0 or len(all_preds) == 1:
                running_acc = accuracy_score(all_labels, all_preds)
                print(f"  [{len(all_preds)}/{len(test_df)}] {video_id} "
                      f"true={CLASS_NAMES[true_label]} pred={CLASS_NAMES[pred_class]} "
                      f"| running_acc={running_acc:.3f} | {speed:.1f} samples/s | "
                      f"ETA: {eta/60:.0f} min")

        except Exception as e:
            print(f"  ERROR {video_id}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time

    # Compute metrics
    print(f"\n{'='*70}")
    print(f"[DaC] Diffusion-as-Classifier Results")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_preds)} | Time: {total_time/60:.1f} min")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"  QWK:       {qwk:.4f}")
    print(f"  MAE:       {mae:.4f}")

    print(f"\n  Per-class report:")
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    print(f"  Confusion Matrix:")
    print(f"  {'':>6} {'  '.join(f'{n:>5}' for n in CLASS_NAMES)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>6} {'  '.join(f'{cm[i,j]:5d}' for j in range(5))}")

    # Prediction distribution
    pred_counts = np.bincount(y_pred, minlength=5)
    true_counts = np.bincount(y_true, minlength=5)
    print(f"\n  Prediction distribution: {dict(zip(CLASS_NAMES, pred_counts.tolist()))}")
    print(f"  True distribution:       {dict(zip(CLASS_NAMES, true_counts.tolist()))}")

    # Save results
    results = {
        "method": "Diffusion-as-Classifier",
        "checkpoint_dir": args.checkpoint_dir,
        "timesteps": args.timesteps,
        "n_noise_samples": args.n_noise_samples,
        "prompt_strategy": args.prompt_strategy,
        "split": args.split,
        "n_samples": len(all_preds),
        "total_time_sec": total_time,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "qwk": float(qwk),
            "mae": float(mae),
        },
        "per_class_recall": {
            CLASS_NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1))
            for i in range(5)
        },
        "confusion_matrix": cm.tolist(),
        "predictions": [
            {
                "video_id": test_df.iloc[i]["video_id"],
                "true": int(all_labels[i]),
                "pred": int(all_preds[i]),
                "losses": all_losses[i],
            }
            for i in range(len(all_preds))
        ],
    }

    results_path = os.path.join(args.output_dir, "dac_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DaC] Results saved to {results_path}")


if __name__ == "__main__":
    main()
