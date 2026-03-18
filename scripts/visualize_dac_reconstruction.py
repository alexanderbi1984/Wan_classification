"""
Visualize DaC reconstruction: decode x_0_pred from each class prompt back to pixel space.

For selected samples, this script:
1. Encodes the video to VAE latent space (x_0)
2. Adds noise at specified timestep to create x_t
3. Runs DiT conditioned on each of the 5 class prompts
4. Recovers x_0_pred = x_t - sigma * v_pred for each prompt
5. Decodes all x_0_pred back through VAE to pixel space
6. Creates a comparison figure: original vs 5 reconstructions

Requires GPU (H100 recommended).

Usage:
    python scripts/visualize_dac_reconstruction.py \
        --checkpoint_dir Wan2.1-T2V-14B \
        --config config_pain/config_lora_t100_aug3.yaml \
        --output_dir results/dac/reconstructions \
        --video_ids "080309_m_29-PA4-057_aligned,080309_m_29-PA1-054_aligned,080309_m_29-PA4-002_aligned" \
        --timesteps 200 500 800
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.online_video import read_frames_from_directory
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

CLASS_NAMES = ["BL1", "PA1", "PA2", "PA3", "PA4"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

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


def load_models(checkpoint_dir, device="cuda"):
    """Load DiT, VAE, and T5."""
    print(f"[Recon] Loading models from {checkpoint_dir}")
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
    print(f"[Recon] Models loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GiB")
    return dit, vae, t5_encoder


def precompute_embeddings(t5_encoder, device):
    embeddings = {}
    for cls, prompts in CLASS_PROMPTS.items():
        embs = t5_encoder(prompts, device)
        max_len = max(e.shape[0] for e in embs)
        padded = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embs])
        embeddings[cls] = padded.mean(dim=0)
    return embeddings


def latent_to_frames(vae, latent):
    """Decode a single latent (C, T, H, W) to a list of RGB numpy frames."""
    with torch.no_grad():
        decoded = vae.decode([latent.clamp(-1, 1)])
    video = decoded[0]  # (C, T, H, W) float32 in [-1, 1]
    video = (video + 1) / 2  # [0, 1]
    video = video.clamp(0, 1)
    frames = []
    for t in range(video.shape[1]):
        frame = video[:, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    return frames


def select_frame_indices(n_frames, n_show=5):
    """Select evenly spaced frame indices for display."""
    return np.linspace(0, n_frames - 1, n_show, dtype=int).tolist()


def create_reconstruction_figure(
    original_frames, recon_frames_per_class, video_id, true_label,
    timestep, sigma, losses, output_path
):
    """Create a comparison figure: original row + 5 reconstruction rows."""
    n_show = 5
    n_orig = len(original_frames)
    n_recon = len(recon_frames_per_class[0])

    orig_indices = select_frame_indices(n_orig, n_show)
    recon_indices = select_frame_indices(n_recon, n_show)

    pred_cls = min(losses, key=losses.get)
    n_rows = 7  # original + 5 classes + loss bar
    fig = plt.figure(figsize=(n_show * 2.5 + 4, n_rows * 2.2))
    gs = gridspec.GridSpec(n_rows, n_show + 2, hspace=0.3, wspace=0.05,
                           width_ratios=[1] * n_show + [0.3, 2])

    # Row 0: original frames
    for col, idx in enumerate(orig_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(original_frames[idx])
        ax.set_axis_off()
        if col == 0:
            ax.set_title("Original", fontsize=9, fontweight="bold", loc="left")

    # Label for original row
    ax_label = fig.add_subplot(gs[0, n_show + 1])
    ax_label.set_axis_off()
    ax_label.text(0, 0.5, f"True: {CLASS_NAMES[true_label]}\n({n_orig} frames)",
                  fontsize=9, va="center", fontweight="bold",
                  color=CLASS_COLORS[true_label])

    # Rows 1-5: reconstructions per class
    for cls in range(5):
        row = cls + 1
        recon_f = recon_frames_per_class[cls]
        for col, idx in enumerate(recon_indices):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(recon_f[idx])
            ax.set_axis_off()
            if cls == pred_cls:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("lime")
                    spine.set_linewidth(3)
            if cls == true_label and cls != pred_cls:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("gold")
                    spine.set_linewidth(2)

        # Label
        ax_label = fig.add_subplot(gs[row, n_show + 1])
        ax_label.set_axis_off()
        loss_val = losses[cls]
        is_winner = cls == pred_cls
        is_true = cls == true_label
        markers = ""
        if is_winner:
            markers += " [PRED]"
        if is_true:
            markers += " [TRUE]"
        ax_label.text(0, 0.5,
                      f"Prompt: {CLASS_NAMES[cls]}{markers}\n"
                      f"Loss: {loss_val:.6f}\n"
                      f"Δ: {loss_val - min(losses.values()):.6f}",
                      fontsize=8, va="center",
                      fontweight="bold" if is_winner else "normal",
                      color=CLASS_COLORS[cls])

    # Row 6: loss bar chart
    ax_bar = fig.add_subplot(gs[n_rows - 1, :])
    loss_vals = [losses[k] for k in range(5)]
    min_loss = min(loss_vals)
    bars = ax_bar.barh(CLASS_NAMES, loss_vals, color=CLASS_COLORS,
                       edgecolor="white", linewidth=0.5, height=0.6)
    for i, (bar, val) in enumerate(zip(bars, loss_vals)):
        delta = val - min_loss
        ax_bar.text(bar.get_width() + (max(loss_vals) - min_loss) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.6f} (Δ={delta:.6f})", va="center", fontsize=7,
                    fontfamily="monospace")
        if i == pred_cls:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
    margin = (max(loss_vals) - min_loss) * 0.6
    ax_bar.set_xlim(min_loss - margin * 0.1, max(loss_vals) + margin)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Reconstruction Loss (MSE)", fontsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)

    correct = pred_cls == true_label
    result_str = "CORRECT" if correct else "WRONG"
    result_color = "#27ae60" if correct else "#c0392b"
    fig.suptitle(
        f"DaC Reconstruction Visualization: {video_id}\n"
        f"Timestep t={timestep:.0f} (σ={sigma:.2f}) | "
        f"True: {CLASS_NAMES[true_label]} → Pred: {CLASS_NAMES[pred_cls]} [{result_str}]",
        fontsize=12, fontweight="bold", color=result_color, y=1.01
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_one_video(dit, vae, class_embeddings, video_id, true_label,
                      frames_root, timesteps, output_dir, resize=128,
                      max_frames=129, sample_rate=2, device="cuda"):
    """Full pipeline: encode → noise → denoise per class → decode → visualize."""
    frame_dir = os.path.join(frames_root, video_id)
    raw_frames = read_frames_from_directory(
        frame_dir, max_frames=max_frames, resize=resize, sample_rate=sample_rate
    )
    video_tensor = torch.stack(raw_frames).permute(1, 0, 2, 3)  # (C, T, H, W)

    # Original frames as numpy for display
    original_frames = [(f.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for f in raw_frames]

    # Encode to latent
    with torch.no_grad():
        latent = vae.encode([video_tensor.to(device)])[0]  # (16, T', H', W')

    _, T_lat, H_lat, W_lat = latent.shape
    seq_len = T_lat * (H_lat // 2) * (W_lat // 2)
    print(f"  {video_id}: latent shape {latent.shape}, seq_len={seq_len}")

    for t_val in timesteps:
        sigma = t_val / 1000.0
        noise = torch.randn_like(latent)
        x_t = (1 - sigma) * latent + sigma * noise

        # Run DiT for all 5 classes (batched)
        x_list = [x_t.clone() for _ in range(5)]
        t_tensor = torch.full((5,), t_val, device=device, dtype=latent.dtype)
        ctx_list = [
            class_embeddings[cls].to(device=device, dtype=latent.dtype)
            for cls in range(5)
        ]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_preds = dit(x_list, t_tensor, ctx_list, seq_len)

        # Recover x_0_pred and compute losses + decode
        recon_frames_per_class = []
        losses = {}
        for cls in range(5):
            x0_pred = x_t - sigma * v_preds[cls]
            losses[cls] = F.mse_loss(x0_pred, latent).item()

            recon_f = latent_to_frames(vae, x0_pred)
            recon_frames_per_class.append(recon_f)

        # Also decode the noisy x_t and original latent for reference
        fname = f"{video_id}_t{int(t_val)}.png"
        create_reconstruction_figure(
            original_frames, recon_frames_per_class,
            video_id, true_label, t_val, sigma, losses,
            os.path.join(output_dir, fname)
        )

        # Free intermediate tensors
        del v_preds, x_list, recon_frames_per_class
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/dac/reconstructions")
    parser.add_argument("--video_ids", type=str, default=None,
                        help="Comma-separated video IDs. If None, auto-selects examples.")
    parser.add_argument("--results_json", type=str, default="results/dac/dac_results.json",
                        help="Previous DaC results to pick examples from")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[200, 500, 800])
    parser.add_argument("--n_auto_examples", type=int, default=2,
                        help="Per-class examples to auto-select (if --video_ids not given)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which videos to visualize
    video_list = []
    if args.video_ids:
        with open(args.results_json) as f:
            results = json.load(f)
        pred_map = {p["video_id"]: p for p in results["predictions"]}
        for vid in args.video_ids.split(","):
            vid = vid.strip()
            if vid in pred_map:
                video_list.append((vid, pred_map[vid]["true"]))
            else:
                print(f"  WARNING: {vid} not found in results, skipping")
    else:
        with open(args.results_json) as f:
            results = json.load(f)
        predictions = results["predictions"]

        # Auto-select: 1 correct + 1 wrong per class (where possible)
        from collections import defaultdict
        correct_by_cls = defaultdict(list)
        wrong_by_cls = defaultdict(list)
        for p in predictions:
            if p["true"] == p["pred"]:
                correct_by_cls[p["true"]].append(p)
            else:
                wrong_by_cls[p["true"]].append(p)

        for cls in range(5):
            for p in correct_by_cls[cls][:1]:
                video_list.append((p["video_id"], p["true"]))
            for p in wrong_by_cls[cls][:1]:
                video_list.append((p["video_id"], p["true"]))

    print(f"[Recon] Will visualize {len(video_list)} videos at timesteps {args.timesteps}")
    for vid, lbl in video_list:
        print(f"  {vid} (true={CLASS_NAMES[lbl]})")

    # Load models
    dit, vae, t5_encoder = load_models(args.checkpoint_dir, args.device)
    class_embeddings = precompute_embeddings(t5_encoder, args.device)

    # Free T5
    t5_encoder.model.cpu()
    del t5_encoder
    torch.cuda.empty_cache()
    print("[Recon] T5 unloaded.")

    # Process each video
    for i, (vid, lbl) in enumerate(video_list):
        print(f"\n[{i+1}/{len(video_list)}] Processing {vid} (true={CLASS_NAMES[lbl]})")
        process_one_video(
            dit, vae, class_embeddings, vid, lbl,
            data_cfg["frames_root"], args.timesteps, args.output_dir,
            resize=data_cfg.get("resize", 128),
            max_frames=data_cfg.get("max_frames", 129),
            sample_rate=data_cfg.get("sample_rate", 2),
            device=args.device,
        )

    print(f"\n[Recon] All done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
