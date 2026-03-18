"""
Visualize Diffusion-as-Classifier (DaC) results.

Part 1: Static analysis from existing results JSON + original frames (no GPU needed).
Creates per-sample loss bar charts alongside original video frames for
correct and incorrect classification examples.

Usage:
    python scripts/visualize_dac_results.py \
        --results results/dac/dac_results.json \
        --config config_pain/config_lora_t100_aug3.yaml \
        --output_dir results/dac/visualizations
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
import yaml

CLASS_NAMES = ["BL1", "PA1", "PA2", "PA3", "PA4"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


def load_frames_thumbnail(frame_dir, indices=(0, -1), resize=128):
    """Load a few representative frames from a video directory."""
    files = sorted(f for f in os.listdir(frame_dir)
                   if f.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")))
    if not files:
        return []
    frames = []
    for idx in indices:
        fname = files[idx]
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize, resize))
            frames.append(img)
    return frames


def plot_single_sample(pred_entry, frames_root, ax_frames, ax_bar, resize=128):
    """Plot original frames + loss bar chart for one sample."""
    video_id = pred_entry["video_id"]
    true_cls = pred_entry["true"]
    pred_cls = pred_entry["pred"]
    losses = {int(k): v for k, v in pred_entry["losses"].items()}

    frame_dir = os.path.join(frames_root, video_id)
    n_frames_to_show = 5
    all_files = sorted(f for f in os.listdir(frame_dir)
                       if f.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")))
    n_total = len(all_files)
    indices = np.linspace(0, n_total - 1, n_frames_to_show, dtype=int)

    strip = []
    for idx in indices:
        img = cv2.imread(os.path.join(frame_dir, all_files[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize, resize))
            strip.append(img)

    if strip:
        combined = np.concatenate(strip, axis=1)
        ax_frames.imshow(combined)
    ax_frames.set_axis_off()
    correct = true_cls == pred_cls
    color = "#27ae60" if correct else "#c0392b"
    symbol = "CORRECT" if correct else "WRONG"
    ax_frames.set_title(
        f"{video_id}\nTrue: {CLASS_NAMES[true_cls]}  |  Pred: {CLASS_NAMES[pred_cls]}  [{symbol}]",
        fontsize=10, fontweight="bold", color=color
    )

    loss_vals = [losses[k] for k in range(5)]
    min_loss = min(loss_vals)
    max_loss = max(loss_vals)

    bars = ax_bar.barh(CLASS_NAMES, loss_vals, color=CLASS_COLORS, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, loss_vals)):
        delta = val - min_loss
        label = f"{val:.5f} (Δ={delta:.5f})"
        ax_bar.text(bar.get_width() + (max_loss - min_loss) * 0.02, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=7, fontfamily="monospace")
        if i == pred_cls:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        if i == true_cls:
            ax_bar.text(bar.get_x() + 0.0001, bar.get_y() + bar.get_height() / 2,
                        "★", va="center", ha="left", fontsize=12, color="black")

    margin = (max_loss - min_loss) * 0.6
    ax_bar.set_xlim(min_loss - margin * 0.1, max_loss + margin)
    ax_bar.set_xlabel("Reconstruction Loss (MSE)", fontsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.invert_yaxis()

    disc = (max_loss - min_loss) / np.mean(loss_vals) * 100
    ax_bar.set_title(f"Per-class loss  |  Range: {max_loss - min_loss:.6f}  |  Discrimination: {disc:.1f}%",
                     fontsize=8)


def create_gallery(results, frames_root, output_path, n_per_category=3):
    """Create a gallery showing correct and incorrect examples per class."""
    predictions = results["predictions"]

    categories = {}
    for cls in range(5):
        correct = [p for p in predictions if p["true"] == cls and p["pred"] == cls]
        wrong = [p for p in predictions if p["true"] == cls and p["pred"] != cls]
        wrong.sort(key=lambda p: abs(
            {int(k): v for k, v in p["losses"].items()}[p["true"]] -
            min(p["losses"].values(), key=float)
        ), reverse=True)
        categories[f"{CLASS_NAMES[cls]}_correct"] = correct[:n_per_category]
        categories[f"{CLASS_NAMES[cls]}_wrong"] = wrong[:n_per_category]

    for cat_name, samples in categories.items():
        if not samples:
            continue
        n = len(samples)
        fig = plt.figure(figsize=(16, 3.5 * n))
        gs = gridspec.GridSpec(n, 2, width_ratios=[2, 1.5], hspace=0.5, wspace=0.3)

        for i, pred_entry in enumerate(samples):
            ax_frames = fig.add_subplot(gs[i, 0])
            ax_bar = fig.add_subplot(gs[i, 1])
            plot_single_sample(pred_entry, frames_root, ax_frames, ax_bar)

        fig.suptitle(f"DaC Results — {cat_name.replace('_', ' ').title()}",
                     fontsize=14, fontweight="bold", y=1.0)
        path = os.path.join(output_path, f"dac_{cat_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")


def create_loss_distribution(results, output_path):
    """Visualize per-class loss distribution across all samples."""
    predictions = results["predictions"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Loss distribution per true class (which class prompt "wins")
    ax = axes[0]
    true_classes = [p["true"] for p in predictions]
    pred_classes = [p["pred"] for p in predictions]
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        cm[t][p] += 1
    im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i][j] > cm.max() * 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2) Mean loss per true class, broken down by prompt class
    ax = axes[1]
    mean_losses = np.zeros((5, 5))
    counts = np.zeros(5)
    for p in predictions:
        t = p["true"]
        counts[t] += 1
        for k, v in p["losses"].items():
            mean_losses[t][int(k)] += v
    for t in range(5):
        if counts[t] > 0:
            mean_losses[t] /= counts[t]

    x = np.arange(5)
    width = 0.15
    for prompt_cls in range(5):
        offset = (prompt_cls - 2) * width
        ax.bar(x + offset, mean_losses[:, prompt_cls], width,
               label=f"Prompt: {CLASS_NAMES[prompt_cls]}", color=CLASS_COLORS[prompt_cls],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Mean Reconstruction Loss")
    ax.set_title("Mean Loss by True Class × Prompt Class")
    ax.legend(fontsize=7, loc="upper left")

    # 3) Discrimination power: how much does the "winner" loss differ from average
    ax = axes[2]
    disc_by_class = {c: [] for c in range(5)}
    for p in predictions:
        losses_v = list(p["losses"].values())
        rng = max(losses_v) - min(losses_v)
        disc = rng / np.mean(losses_v) * 100
        disc_by_class[p["true"]].append(disc)

    bp_data = [disc_by_class[c] for c in range(5)]
    bp = ax.boxplot(bp_data, labels=CLASS_NAMES, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Discrimination (%)")
    ax.set_title("Loss Discrimination Power by True Class\n(max-min / mean × 100)")
    ax.axhline(y=np.mean([np.mean(v) for v in disc_by_class.values()]),
               color="red", linestyle="--", alpha=0.5, label="Overall mean")
    ax.legend(fontsize=8)

    fig.suptitle(f"DaC Loss Analysis  |  Acc={results['metrics']['accuracy']:.3f}  "
                 f"QWK={results['metrics']['qwk']:.3f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_path, "dac_loss_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def create_loss_margin_analysis(results, output_path):
    """Analyze: does the model even use prompts differently?"""
    predictions = results["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Scatter: absolute loss vs discrimination ratio
    ax = axes[0]
    for p in predictions:
        losses_v = [float(v) for v in p["losses"].values()]
        mean_l = np.mean(losses_v)
        disc = (max(losses_v) - min(losses_v)) / mean_l * 100
        correct = p["true"] == p["pred"]
        ax.scatter(mean_l, disc,
                   c=CLASS_COLORS[p["true"]], alpha=0.3, s=15,
                   marker="o" if correct else "x", linewidths=0.5)
    ax.set_xlabel("Mean Reconstruction Loss (all prompts)")
    ax.set_ylabel("Discrimination Ratio (%)")
    ax.set_title("Loss Magnitude vs Discrimination\n(○ = correct, × = wrong)")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=CLASS_COLORS[i], markersize=8,
                              label=CLASS_NAMES[i]) for i in range(5)]
    ax.legend(handles=legend_elements, fontsize=8)

    # 2) Per-prompt average loss (does any prompt systematically produce lower loss?)
    ax = axes[1]
    prompt_losses = {c: [] for c in range(5)}
    for p in predictions:
        for k, v in p["losses"].items():
            prompt_losses[int(k)].append(v)

    means = [np.mean(prompt_losses[c]) for c in range(5)]
    stds = [np.std(prompt_losses[c]) for c in range(5)]
    bars = ax.bar(CLASS_NAMES, means, yerr=stds, color=CLASS_COLORS,
                  edgecolor="white", capsize=4, alpha=0.85)
    ax.set_ylabel("Mean Reconstruction Loss")
    ax.set_title("Average Loss per Prompt Class (across ALL samples)\n"
                 "→ Lower = model finds this prompt 'easier'")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{m:.5f}", ha="center", va="bottom", fontsize=8, fontfamily="monospace")

    ymin = min(means) - max(stds) * 1.5
    ymax = max(means) + max(stds) * 2
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    path = os.path.join(output_path, "dac_margin_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/dac/dac_results.json")
    parser.add_argument("--config", type=str, default="config_pain/config_lora_t100_aug3.yaml")
    parser.add_argument("--output_dir", type=str, default="results/dac/visualizations")
    parser.add_argument("--n_examples", type=int, default=3)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    frames_root = config["data"]["frames_root"]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Vis] Loaded {len(results['predictions'])} predictions")
    print(f"[Vis] Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"[Vis] Output: {args.output_dir}")

    print("\n[1/3] Creating loss distribution analysis...")
    create_loss_distribution(results, args.output_dir)

    print("\n[2/3] Creating loss margin analysis...")
    create_loss_margin_analysis(results, args.output_dir)

    print("\n[3/3] Creating per-sample galleries...")
    create_gallery(results, frames_root, args.output_dir, n_per_category=args.n_examples)

    print(f"\n[Vis] All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
