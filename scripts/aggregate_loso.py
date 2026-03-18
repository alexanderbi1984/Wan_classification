"""
Aggregate LOSO fold results into a single summary.

Usage:
    python scripts/aggregate_loso.py --results_dir results/loso
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Aggregate LOSO results")
    parser.add_argument("--results_dir", type=str, default="results/loso")
    args = parser.parse_args()

    result_files = sorted(
        f for f in os.listdir(args.results_dir)
        if f.endswith(".json") and f != "loso_summary.json"
    )

    if not result_files:
        print(f"No result files found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} fold results\n")

    # Collect per-fold metrics
    all_metrics = defaultdict(list)
    all_results = []

    # For pooled accuracy computation
    total_correct = 0
    total_samples = 0
    per_class_correct = defaultdict(float)
    per_class_total = defaultdict(int)

    for fname in result_files:
        with open(os.path.join(args.results_dir, fname)) as f:
            fold = json.load(f)

        subject = fold["test_subject"]
        metrics = fold.get("metrics", {})
        true_counts = fold.get("true_counts_per_class", {})
        n_test = fold.get("n_test_samples", 0)

        all_results.append(fold)

        # Collect scalar metrics
        for key in ["test_pain_QWK", "test_pain_Accuracy", "test_pain_F1", "test_pain_MAE"]:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        for c in range(5):
            recall_key = f"test_pain_recall_c{c}"
            pred_key = f"test_pain_pred_count_c{c}"
            if recall_key in metrics:
                all_metrics[recall_key].append(metrics[recall_key])

            # Pooled accuracy: correct_c = recall_c * true_count_c
            true_c = true_counts.get(str(c), true_counts.get(c, 0))
            if recall_key in metrics and true_c > 0:
                correct_c = metrics[recall_key] * true_c
                per_class_correct[c] += correct_c
                per_class_total[c] += true_c

        total_samples += n_test

    total_correct = sum(per_class_correct.values())

    # Compute aggregated metrics
    print("=" * 70)
    print(f"  LOSO Results — {len(result_files)} / 87 folds")
    print("=" * 70)

    print("\n--- Macro-Averaged Metrics (mean ± std across folds) ---\n")
    for key in ["test_pain_QWK", "test_pain_Accuracy", "test_pain_F1", "test_pain_MAE"]:
        vals = all_metrics.get(key, [])
        if vals:
            print(f"  {key:30s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")

    print("\n--- Per-Class Recall (macro-averaged across folds) ---\n")
    for c in range(5):
        key = f"test_pain_recall_c{c}"
        vals = all_metrics.get(key, [])
        if vals:
            print(f"  c{c}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n--- Pooled (Micro) Accuracy ---\n")
    if total_samples > 0:
        pooled_acc = total_correct / total_samples
        print(f"  Pooled accuracy: {pooled_acc:.4f}  ({int(total_correct)}/{total_samples})")
    print(f"\n--- Pooled Per-Class Recall ---\n")
    for c in range(5):
        if per_class_total[c] > 0:
            recall = per_class_correct[c] / per_class_total[c]
            print(f"  c{c}: {recall:.4f}  ({per_class_correct[c]:.0f}/{per_class_total[c]})")

    # Per-fold detail table
    print(f"\n--- Per-Fold Detail ---\n")
    print(f"{'Subject':>15s}  {'QWK':>6s}  {'Acc':>6s}  {'F1':>6s}  {'MAE':>6s}  {'c0':>5s}  {'c1':>5s}  {'c2':>5s}  {'c3':>5s}  {'c4':>5s}")
    print("-" * 90)
    for fold in all_results:
        m = fold.get("metrics", {})
        subj = fold["test_subject"]
        qwk = m.get("test_pain_QWK", float("nan"))
        acc = m.get("test_pain_Accuracy", float("nan"))
        f1 = m.get("test_pain_F1", float("nan"))
        mae = m.get("test_pain_MAE", float("nan"))
        recalls = [m.get(f"test_pain_recall_c{c}", float("nan")) for c in range(5)]
        print(f"{subj:>15s}  {qwk:6.3f}  {acc:6.3f}  {f1:6.3f}  {mae:6.3f}  " +
              "  ".join(f"{r:5.3f}" for r in recalls))

    # Save summary JSON
    summary = {
        "n_folds": len(result_files),
        "n_total_subjects": 87,
        "macro_mean": {k: float(np.mean(v)) for k, v in all_metrics.items()},
        "macro_std": {k: float(np.std(v)) for k, v in all_metrics.items()},
        "pooled_accuracy": float(total_correct / total_samples) if total_samples > 0 else None,
        "pooled_per_class_recall": {
            str(c): float(per_class_correct[c] / per_class_total[c])
            for c in range(5) if per_class_total[c] > 0
        },
        "total_samples": total_samples,
    }
    summary_path = os.path.join(args.results_dir, "loso_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
