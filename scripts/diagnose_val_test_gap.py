"""Diagnose val-test gap: per-subject, gender-stratified, confusion matrices.

Runs inference on both val and test sets using a trained checkpoint,
then produces detailed breakdowns to identify the root cause of the
val-test performance gap.

Usage (4 GPUs via torchrun):
    torchrun --standalone --nproc_per_node=4 scripts/diagnose_val_test_gap.py \
        --config config_pain/config_lora_t100_aug2.yaml \
        --checkpoint path/to/best.ckpt
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    recall_score,
    mean_absolute_error,
    confusion_matrix,
)
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.online_multimodal import BioVidOnlineClassifier
from data.online_video import BioVidOnlineDataset, collate_biovid_online


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def log(msg):
    if rank() == 0:
        print(msg, flush=True)


def coral_prob_to_label(logits, num_classes=5):
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).clamp(max=num_classes - 1)


def ce_logits_to_label(logits):
    return logits.argmax(dim=1)


def gather_variable_tensors(local_tensor, world_sz):
    if world_sz == 1:
        return local_tensor

    local_size = torch.tensor([local_tensor.shape[0]], device=local_tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_sz)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    padded = torch.zeros(max_size, *local_tensor.shape[1:],
                         device=local_tensor.device, dtype=local_tensor.dtype)
    padded[:local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_sz)]
    dist.all_gather(gathered, padded)

    result = []
    for i, sz in enumerate(all_sizes):
        result.append(gathered[i][:sz.item()])
    return torch.cat(result, dim=0)


def compute_metrics(gt, pred, num_classes=5):
    return {
        "QWK": cohen_kappa_score(gt, pred, weights="quadratic"),
        "Acc": accuracy_score(gt, pred),
        "F1": f1_score(gt, pred, average="macro", zero_division=0),
        "MAE": mean_absolute_error(gt, pred),
        "per_class_recall": recall_score(
            gt, pred, average=None,
            labels=list(range(num_classes)), zero_division=0,
        ),
    }


def run_inference(model, loader, device, num_classes, eval_head):
    """Run inference and return (predictions, labels, video_ids) as CPU tensors."""
    all_preds = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (video, labels) in enumerate(loader):
            video = video.to(device)
            out = model(video)

            if eval_head == "ce" and "pain_ce" in out:
                preds = ce_logits_to_label(out["pain_ce"])
            else:
                preds = coral_prob_to_label(out["pain_coral"], num_classes)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

            if rank() == 0 and ((batch_idx + 1) % 10 == 0 or batch_idx == 0):
                print(f"  Batch {batch_idx + 1}/{len(loader)}", flush=True)

    return torch.cat(all_preds), torch.cat(all_labels)


def print_confusion_matrix(gt, pred, num_classes, title):
    cm = confusion_matrix(gt, pred, labels=list(range(num_classes)))
    class_names = ["BL1", "PA1", "PA2", "PA3", "PA4"]
    log(f"\n{'=' * 60}")
    log(f"Confusion Matrix: {title}")
    log(f"{'=' * 60}")
    header = "Pred->  " + "  ".join(f"{c:>5s}" for c in class_names[:num_classes])
    log(header)
    log("-" * len(header))
    for i in range(num_classes):
        row = f"{class_names[i]:>5s}   " + "  ".join(f"{cm[i, j]:5d}" for j in range(num_classes))
        total = cm[i].sum()
        pct = cm[i, i] / total * 100 if total > 0 else 0
        row += f"  | {pct:5.1f}% ({total})"
        log(row)


def print_prediction_distribution(gt, pred, num_classes, title):
    log(f"\n{'=' * 60}")
    log(f"Prediction Distribution: {title}")
    log(f"{'=' * 60}")
    class_names = ["BL1", "PA1", "PA2", "PA3", "PA4"]
    gt_counts = np.bincount(gt, minlength=num_classes)
    pred_counts = np.bincount(pred, minlength=num_classes)

    log(f"{'Class':<8s} {'True':>8s} {'Pred':>8s} {'Ratio':>8s}")
    log("-" * 35)
    for i in range(num_classes):
        ratio = pred_counts[i] / gt_counts[i] if gt_counts[i] > 0 else 0
        log(f"{class_names[i]:<8s} {gt_counts[i]:8d} {pred_counts[i]:8d} {ratio:8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose val-test gap")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="Override labels CSV (default: from config)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        log(f"[Diag] DDP initialized: {world_size()} GPUs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log("[Diag] Single-GPU mode")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = config.get("model_params", {})

    labels_csv = args.labels_csv or data_cfg["labels_csv"]
    eval_head = model_cfg.get("eval_head", "coral")
    num_classes = data_cfg.get("num_classes", 5)

    log(f"[Diag] Loading checkpoint: {args.checkpoint}")
    model = BioVidOnlineClassifier.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    labels_df = pd.read_csv(labels_csv)

    results = {}
    for split in ["val", "test"]:
        log(f"\n{'#' * 60}")
        log(f"  Running inference on {split.upper()} set")
        log(f"{'#' * 60}")

        ds = BioVidOnlineDataset(
            split=split,
            labels_csv=labels_csv,
            frames_root=data_cfg["frames_root"],
            num_classes=num_classes,
            resize=data_cfg.get("resize", 128),
            max_frames=data_cfg.get("max_frames", 129),
            sample_rate=data_cfg.get("sample_rate", 2),
        )

        sampler = DistributedSampler(ds, shuffle=False) if is_dist() else None
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_biovid_online,
            pin_memory=True, drop_last=False,
        )

        preds, labels = run_inference(model, loader, device, num_classes, eval_head)

        ws = world_size()
        if ws > 1:
            preds = gather_variable_tensors(preds.to(device), ws).cpu()
            labels = gather_variable_tensors(labels.to(device), ws).cpu()
            if is_dist():
                dist.barrier()

        results[split] = {
            "preds": preds.numpy(),
            "labels": labels.numpy(),
            "video_ids": [s["video_id"] for s in ds.samples],
        }

    if rank() != 0:
        if is_dist():
            dist.destroy_process_group()
        return

    # ================================================================
    # ANALYSIS (rank 0 only)
    # ================================================================

    for split in ["val", "test"]:
        gt = results[split]["labels"]
        pred = results[split]["preds"]
        video_ids = results[split]["video_ids"]

        # Trim gathered results to dataset size
        n = len(video_ids)
        gt = gt[:n]
        pred = pred[:n]

        results[split]["labels"] = gt
        results[split]["preds"] = pred

        metrics = compute_metrics(gt, pred, num_classes)
        log(f"\n{'=' * 60}")
        log(f"Overall Metrics: {split.upper()} (N={n})")
        log(f"{'=' * 60}")
        log(f"  QWK  = {metrics['QWK']:.4f}")
        log(f"  Acc  = {metrics['Acc']:.4f}")
        log(f"  F1   = {metrics['F1']:.4f}")
        log(f"  MAE  = {metrics['MAE']:.4f}")
        recall = metrics["per_class_recall"]
        log(f"  Per-class recall: " +
            " | ".join(f"c{i}={recall[i]:.3f}" for i in range(num_classes)))

        print_confusion_matrix(gt, pred, num_classes, split.upper())
        print_prediction_distribution(gt, pred, num_classes, split.upper())

    # ================================================================
    # PER-SUBJECT ANALYSIS
    # ================================================================
    log(f"\n{'#' * 60}")
    log(f"  PER-SUBJECT ANALYSIS")
    log(f"{'#' * 60}")

    for split in ["val", "test"]:
        gt = results[split]["labels"]
        pred = results[split]["preds"]
        video_ids = results[split]["video_ids"]

        split_df = labels_df[labels_df["split"].str.lower() == split].copy()
        vid_to_subject = dict(zip(split_df["video_id"], split_df["subject_id"]))
        vid_to_sex = dict(zip(split_df["video_id"], split_df["sex"]))
        vid_to_age = dict(zip(split_df["video_id"], split_df["age"]))

        subject_results = {}
        for i, vid in enumerate(video_ids):
            if i >= len(gt):
                break
            subj = vid_to_subject.get(vid, "unknown")
            if subj not in subject_results:
                subject_results[subj] = {
                    "gt": [], "pred": [],
                    "sex": vid_to_sex.get(vid, "?"),
                    "age": vid_to_age.get(vid, "?"),
                }
            subject_results[subj]["gt"].append(gt[i])
            subject_results[subj]["pred"].append(pred[i])

        log(f"\n{'=' * 60}")
        log(f"Per-Subject Breakdown: {split.upper()}")
        log(f"{'=' * 60}")
        log(f"{'Subject':<20s} {'Sex':>3s} {'Age':>4s} {'N':>5s} "
            f"{'QWK':>7s} {'Acc':>7s} {'F1':>7s} {'MAE':>7s}")
        log("-" * 75)

        subject_rows = []
        for subj in sorted(subject_results.keys()):
            data = subject_results[subj]
            s_gt = np.array(data["gt"])
            s_pred = np.array(data["pred"])
            n = len(s_gt)

            if len(np.unique(s_gt)) < 2:
                qwk = float("nan")
            else:
                qwk = cohen_kappa_score(s_gt, s_pred, weights="quadratic")
            acc = accuracy_score(s_gt, s_pred)
            f1 = f1_score(s_gt, s_pred, average="macro", zero_division=0)
            mae = mean_absolute_error(s_gt, s_pred)

            sex = data["sex"]
            age = data["age"]
            log(f"{subj:<20s} {sex:>3s} {age:>4} {n:5d} "
                f"{qwk:7.3f} {acc:7.3f} {f1:7.3f} {mae:7.3f}")

            subject_rows.append({
                "subject": subj, "sex": sex, "age": age,
                "n": n, "qwk": qwk, "acc": acc, "f1": f1, "mae": mae,
            })

        # Subject-level statistics
        accs = [r["acc"] for r in subject_rows]
        maes = [r["mae"] for r in subject_rows]
        log(f"\n  Subject Acc: mean={np.mean(accs):.3f}, "
            f"std={np.std(accs):.3f}, "
            f"min={np.min(accs):.3f}, max={np.max(accs):.3f}")
        log(f"  Subject MAE: mean={np.mean(maes):.3f}, "
            f"std={np.std(maes):.3f}, "
            f"min={np.min(maes):.3f}, max={np.max(maes):.3f}")

    # ================================================================
    # GENDER-STRATIFIED ANALYSIS
    # ================================================================
    log(f"\n{'#' * 60}")
    log(f"  GENDER-STRATIFIED ANALYSIS")
    log(f"{'#' * 60}")

    for split in ["val", "test"]:
        gt = results[split]["labels"]
        pred = results[split]["preds"]
        video_ids = results[split]["video_ids"]

        split_df = labels_df[labels_df["split"].str.lower() == split].copy()
        vid_to_sex = dict(zip(split_df["video_id"], split_df["sex"]))

        for sex_label, sex_name in [("m", "Male"), ("w", "Female")]:
            mask = np.array([
                vid_to_sex.get(video_ids[i], "?") == sex_label
                for i in range(len(gt))
            ])
            if mask.sum() == 0:
                continue

            s_gt = gt[mask]
            s_pred = pred[mask]
            metrics = compute_metrics(s_gt, s_pred, num_classes)

            recall = metrics["per_class_recall"]
            log(f"\n  {split.upper()} - {sex_name} (N={mask.sum()}):")
            log(f"    QWK={metrics['QWK']:.4f}  Acc={metrics['Acc']:.4f}  "
                f"F1={metrics['F1']:.4f}  MAE={metrics['MAE']:.4f}")
            log(f"    Recall: " +
                " | ".join(f"c{i}={recall[i]:.3f}" for i in range(num_classes)))

    # ================================================================
    # VAL vs TEST GAP SUMMARY
    # ================================================================
    log(f"\n{'#' * 60}")
    log(f"  VAL vs TEST GAP SUMMARY")
    log(f"{'#' * 60}")

    val_m = compute_metrics(results["val"]["labels"], results["val"]["preds"], num_classes)
    test_m = compute_metrics(results["test"]["labels"], results["test"]["preds"], num_classes)

    log(f"\n{'Metric':<12s} {'Val':>8s} {'Test':>8s} {'Gap':>8s}")
    log("-" * 40)
    for key in ["QWK", "Acc", "F1", "MAE"]:
        v = val_m[key]
        t = test_m[key]
        gap = v - t if key != "MAE" else t - v
        log(f"{key:<12s} {v:8.4f} {t:8.4f} {gap:+8.4f}")

    log(f"\n{'Class':<8s} {'Val Recall':>10s} {'Test Recall':>11s} {'Gap':>8s}")
    log("-" * 40)
    class_names = ["BL1", "PA1", "PA2", "PA3", "PA4"]
    for i in range(num_classes):
        vr = val_m["per_class_recall"][i]
        tr = test_m["per_class_recall"][i]
        log(f"{class_names[i]:<8s} {vr:10.3f} {tr:11.3f} {vr - tr:+8.3f}")

    log(f"\n{'=' * 60}")
    log("Diagnosis complete.")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
