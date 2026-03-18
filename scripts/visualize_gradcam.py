"""
Grad-CAM visualization for Wan DiT + LoRA pain classifier.

Generates spatial heatmaps showing which regions of the input video
the model attends to when making pain classification predictions.

Usage:
    python scripts/visualize_gradcam.py \
        --config config_pain/config_lora_t100_aug3.yaml \
        --checkpoint results/.../best.ckpt \
        --output_dir results/gradcam \
        --n_samples 5 \
        --target_classes 0 1 2 3 4

Architecture flow for Grad-CAM:
    video (B,C,T,H,W) -> VAE -> vae_latents (B,16,T',H',W')
                               -> DiT -> dit_features (B,dim,T',H',W')  <-- HOOK HERE
                               -> XDiTProcessor(spatial_mean) -> temporal -> pool -> head -> logits
    Gradient:  target_logit -> backprop -> grad w.r.t. dit_features
    Grad-CAM:  channel_weights = GAP(grad)  ;  cam = ReLU(sum(w * feat))
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from classifier.online_multimodal import BioVidOnlineClassifier
from data.online_video import read_frames_from_directory
from train_online_wan import build_model


ALL_CLASS_NAMES = ["BL1 (no pain)", "PA1 (low)", "PA2 (medium)", "PA3 (high)", "PA4 (very high)"]
COLORMAP = cv2.COLORMAP_JET

# Will be set in main() based on config; defaults to 5-class
CLASS_NAMES = ALL_CLASS_NAMES
CLASS_SUBSET = None  # e.g., [0, 4] for binary BL1 vs PA4


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint in float32 (requires >=80GB GPU like H100)."""
    print(f"[GradCAM] Loading checkpoint: {checkpoint_path}")
    model = BioVidOnlineClassifier.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    # GRU backward requires training mode for cuDNN; safe since num_layers=1 (no dropout effect)
    model.temporal_encoder.train()
    model.to(device)
    print(f"[GradCAM] Model loaded on {device} (float32)")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GradCAM] VRAM: {alloc:.1f} / {total:.1f} GiB")
    return model


def load_sample(frames_root, video_id, resize=128, max_frames=129, sample_rate=2):
    """Load a single video sample and return both raw frames and model input tensor."""
    frame_dir = os.path.join(frames_root, video_id)
    frames = read_frames_from_directory(
        frame_dir, max_frames=max_frames, resize=resize, sample_rate=sample_rate
    )
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
    return video_tensor, frames


class GradCAMExtractor:
    """Extract Grad-CAM heatmaps from the DiT feature extraction stage.

    Hooks into the model's forward pass to capture dit_features and their
    gradients, then computes spatial attribution maps.
    """

    def __init__(self, model):
        self.model = model
        self._features = None
        self._gradients = None

    def _forward_with_hook(self, video_tensor):
        """Run forward pass, capturing dit_features and enabling gradient flow."""
        model = self.model

        import torch.cuda.amp as amp

        vae_scale = [model._vae_scale_mean, model._vae_scale_inv_std]
        with torch.no_grad(), amp.autocast(dtype=model._vae_dtype):
            vae_latents = torch.stack([
                model.vae_model.encode(
                    video_tensor[i].unsqueeze(0), vae_scale
                ).float().squeeze(0)
                for i in range(video_tensor.shape[0])
            ])

        # DiT features WITH gradients
        dit_features = model._extract_dit_features(vae_latents)
        dit_features.retain_grad()
        self._features = dit_features

        # Continue downstream pipeline
        vae_feat = model.vae_processor(vae_latents)
        xdit_feat = model.xdit_processor(dit_features)

        if vae_feat.shape[1] != xdit_feat.shape[1]:
            vae_feat = F.interpolate(
                vae_feat.transpose(1, 2), size=xdit_feat.shape[1], mode="nearest"
            ).transpose(1, 2)

        fused = model.fusion(vae_feat, xdit_feat)
        encoded = model.temporal_encoder(fused)
        pooled = model.temporal_pooling(encoded)
        shared = model.shared_encoder(pooled)

        out = {"pain_coral": model.pain_head(shared)}
        if model._use_ce:
            out["pain_ce"] = model.ce_pain_head(shared)
        return out, vae_latents

    def generate(self, video_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a single video.

        Args:
            video_tensor: (1, C, T, H, W) input video.
            target_class: Class index to compute gradient for.
                          If None, uses the predicted class.

        Returns:
            cam: (T', H', W') numpy array, normalized to [0, 1].
            pred_class: Predicted class index.
            pred_probs: Per-class probabilities.
            grid_t, grid_h, grid_w: Spatial grid dimensions.
        """
        self.model.zero_grad()

        out, vae_latents = self._forward_with_hook(video_tensor)

        # Get prediction
        eval_head = self.model.hparams.get("eval_head", "coral")
        if eval_head == "ce" and "pain_ce" in out:
            logits = out["pain_ce"]
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
        else:
            logits = out["pain_coral"]
            cum_probs = torch.sigmoid(logits)
            pred_class = (cum_probs > 0.5).sum(dim=1).clamp(
                max=self.model.hparams.num_pain_classes - 1
            ).item()
            # Convert CORAL cumulative probs to per-class probs
            n_cls = self.model.hparams.num_pain_classes
            probs = torch.zeros(1, n_cls, device=logits.device)
            cp = torch.cat([
                torch.ones(1, 1, device=logits.device), cum_probs,
                torch.zeros(1, 1, device=logits.device)
            ], dim=1)
            for k in range(n_cls):
                probs[0, k] = cp[0, k] - cp[0, k + 1]
            probs = probs.clamp(min=0)

        if target_class is None:
            target_class = pred_class

        # Backpropagate from target class score
        if eval_head == "ce" and "pain_ce" in out:
            score = out["pain_ce"][0, target_class]
        else:
            # For CORAL: use sum of sigmoid logits up to target class
            if target_class == 0:
                score = -torch.sigmoid(logits[0, 0])
            else:
                score = torch.sigmoid(logits[0, :target_class]).sum()

        score.backward(retain_graph=False)

        # Compute Grad-CAM
        grads = self._features.grad  # (1, dim, T', H', W')
        feats = self._features.detach()  # (1, dim, T', H', W')

        # Channel weights via global average pooling of gradients
        weights = grads.mean(dim=[2, 3, 4], keepdim=True)  # (1, dim, 1, 1, 1)

        # Weighted combination
        cam = (weights * feats).sum(dim=1, keepdim=False)  # (1, T', H', W')
        cam = F.relu(cam)  # Only positive contributions

        cam = cam[0].cpu().numpy()  # (T', H', W')

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        probs_np = probs[0].detach().cpu().numpy()

        return cam, pred_class, probs_np


def overlay_heatmap(frame_rgb, heatmap_2d, alpha=0.5):
    """Overlay a heatmap on an RGB frame.

    Args:
        frame_rgb: (H, W, 3) uint8 numpy array.
        heatmap_2d: (H, W) float array in [0, 1].
        alpha: Blending factor.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    heatmap_uint8 = (heatmap_2d * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, COLORMAP)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color.astype(np.float32)
               + (1 - alpha) * frame_rgb.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def visualize_sample(
    model, extractor, video_id, true_label, frames_root, output_dir,
    resize=128, max_frames=129, sample_rate=2, n_display_frames=8, device="cuda"
):
    """Generate and save Grad-CAM visualization for one video sample."""
    video_tensor, raw_frames = load_sample(
        frames_root, video_id, resize, max_frames, sample_rate
    )
    video_input = video_tensor.unsqueeze(0).to(device)

    cam, pred_class, probs = extractor.generate(video_input, target_class=None)
    T_cam, H_cam, W_cam = cam.shape

    # Select evenly-spaced frames for display
    n_orig = len(raw_frames)
    # Map cam temporal indices back to original frame indices
    # VAE temporal downsampling factor
    t_ratio = n_orig / T_cam
    display_cam_indices = np.linspace(0, T_cam - 1, min(n_display_frames, T_cam), dtype=int)

    # Temporal-averaged heatmap for summary column
    cam_avg = cam.mean(axis=0)  # (H_cam, W_cam)
    cam_avg_up = cv2.resize(cam_avg, (resize, resize), interpolation=cv2.INTER_LINEAR)
    cam_avg_up = cam_avg_up / (cam_avg_up.max() + 1e-8)

    n_cols = len(display_cam_indices) + 1  # +1 for temporal average
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    correct_str = "correct" if pred_class == true_label else "wrong"
    fig.suptitle(
        f"{video_id}  |  True: {CLASS_NAMES[true_label]}  |  Pred: {CLASS_NAMES[pred_class]}  "
        f"[{correct_str.upper()}]\n"
        f"Probs: {' '.join(f'c{i}={probs[i]:.2f}' for i in range(len(probs)))}",
        fontsize=12, fontweight="bold",
        color="green" if pred_class == true_label else "red"
    )

    for col, t_cam in enumerate(display_cam_indices):
        t_orig = min(int(t_cam * t_ratio), n_orig - 1)
        frame_np = (raw_frames[t_orig].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        hm = cam[t_cam]
        hm_up = cv2.resize(hm, (resize, resize), interpolation=cv2.INTER_LINEAR)

        axes[0, col].imshow(frame_np)
        axes[0, col].set_title(f"Frame {t_orig}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(hm_up, cmap="jet", vmin=0, vmax=1)
        axes[1, col].set_title(f"Heatmap t={t_cam}", fontsize=9)
        axes[1, col].axis("off")

        blended = overlay_heatmap(frame_np, hm_up, alpha=0.45)
        axes[2, col].imshow(blended)
        axes[2, col].set_title("Overlay", fontsize=9)
        axes[2, col].axis("off")

    # Last column: temporal average
    mid_orig = min(int((T_cam // 2) * t_ratio), n_orig - 1)
    mid_frame = (raw_frames[mid_orig].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    axes[0, -1].imshow(mid_frame)
    axes[0, -1].set_title("Mid frame", fontsize=9)
    axes[0, -1].axis("off")

    axes[1, -1].imshow(cam_avg_up, cmap="jet", vmin=0, vmax=1)
    axes[1, -1].set_title("Temporal Avg", fontsize=9, fontweight="bold")
    axes[1, -1].axis("off")

    blended_avg = overlay_heatmap(mid_frame, cam_avg_up, alpha=0.45)
    axes[2, -1].imshow(blended_avg)
    axes[2, -1].set_title("Avg Overlay", fontsize=9, fontweight="bold")
    axes[2, -1].axis("off")

    plt.tight_layout()
    # Organize into correct/wrong subdirectories
    sub_dir = os.path.join(output_dir, correct_str)
    os.makedirs(sub_dir, exist_ok=True)
    fname = f"{video_id}_true{true_label}_pred{pred_class}.png"
    save_path = os.path.join(sub_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GradCAM] Saved: {save_path}")

    return {
        "video_id": video_id,
        "true_label": true_label,
        "pred_class": pred_class,
        "correct": pred_class == true_label,
        "probs": probs.tolist(),
    }


def select_samples(labels_csv, n_per_class=3, split="test", seed=123,
                   class_subset=None):
    """Select a balanced set of test samples across pain levels.

    Args:
        class_subset: If set (e.g. [0,4]), only select from these pain_levels
                      and remap labels to 0,1,2,... for the model.
    """
    df = pd.read_csv(labels_csv)
    df = df[df["split"] == split]

    if class_subset is not None:
        df = df[df["pain_level"].isin(class_subset)]

    selected = []
    for cls in sorted(df["pain_level"].unique()):
        cls_df = df[df["pain_level"] == cls]
        n = min(n_per_class, len(cls_df))
        sampled = cls_df.sample(n=n, random_state=seed)
        for _, row in sampled.iterrows():
            model_label = cls
            if class_subset is not None:
                model_label = class_subset.index(cls)
            selected.append({
                "video_id": row["video_id"],
                "label": model_label,
                "pain_level": int(row["pain_level"]),
            })

    return selected


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for Wan classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/gradcam")
    parser.add_argument("--n_per_class", type=int, default=3,
                        help="Number of samples per pain class to visualize")
    parser.add_argument("--target_classes", type=int, nargs="*", default=None,
                        help="Only visualize specific classes (default: all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to sample from")
    parser.add_argument("--n_display_frames", type=int, default=8,
                        help="Number of frames to display per sample")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle binary / subset class mapping
    global CLASS_NAMES, CLASS_SUBSET
    class_subset = data_cfg.get("class_subset", None)
    if class_subset is not None:
        CLASS_SUBSET = class_subset
        CLASS_NAMES = [ALL_CLASS_NAMES[i] for i in class_subset]
        print(f"[GradCAM] Binary/subset mode: {CLASS_NAMES}")
    else:
        CLASS_NAMES = ALL_CLASS_NAMES

    # Load model
    model = load_model(args.checkpoint, device=args.device)
    extractor = GradCAMExtractor(model)

    # Select samples
    samples = select_samples(
        data_cfg["labels_csv"],
        n_per_class=args.n_per_class,
        split=args.split,
        class_subset=class_subset,
    )

    if args.target_classes is not None:
        samples = [s for s in samples if s["pain_level"] in args.target_classes]

    print(f"[GradCAM] Visualizing {len(samples)} samples "
          f"({args.n_per_class} per class, split={args.split})")

    results = []
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing {sample['video_id']} "
              f"(class {sample['label']}: {CLASS_NAMES[sample['label']]})")
        try:
            res = visualize_sample(
                model, extractor,
                video_id=sample["video_id"],
                true_label=sample["label"],
                frames_root=data_cfg["frames_root"],
                output_dir=args.output_dir,
                resize=data_cfg.get("resize", 128),
                max_frames=data_cfg.get("max_frames", 129),
                sample_rate=data_cfg.get("sample_rate", 2),
                n_display_frames=args.n_display_frames,
                device=args.device,
            )
            results.append(res)
        except Exception as e:
            print(f"[GradCAM] ERROR processing {sample['video_id']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"[GradCAM] Summary: {len(results)}/{len(samples)} samples processed")
    correct = sum(1 for r in results if r["correct"])
    print(f"[GradCAM] Accuracy on visualized samples: {correct}/{len(results)}")
    print(f"[GradCAM] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
