"""Test a prompt-conditioned checkpoint with prompt ensemble (DDP multi-GPU).

For each test sample, run inference with every prompt in the pool,
then aggregate predictions via:
  - Soft voting: average CORAL cumulative probabilities, then decode label
  - Hard voting: majority vote over per-prompt predicted labels

Supports multi-GPU via torchrun for practical runtime on 14B models.

Usage (single GPU):
    python scripts/test_prompt_ensemble.py \
        --config config_pain/config_lora_dim256_prompt.yaml \
        --checkpoint path/to/best.ckpt \
        --prompt_embeddings prompt_embeddings.pt

Usage (4 GPUs via torchrun):
    torchrun --standalone --nproc_per_node=4 scripts/test_prompt_ensemble.py \
        --config config_pain/config_lora_dim256_prompt.yaml \
        --checkpoint path/to/best.ckpt \
        --prompt_embeddings prompt_embeddings.pt
"""

import argparse
import os
import sys
import time
from collections import Counter

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score
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


def coral_prob_to_label(probs, num_classes=5):
    """Convert CORAL cumulative probabilities to class labels."""
    labels = (probs > 0.5).sum(dim=1)
    return labels.clamp(max=num_classes - 1)


def gather_tensors(tensor, world_sz):
    """All-gather tensors from all ranks into a single concatenated tensor on rank 0."""
    if world_sz == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_sz)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def gather_variable_tensors(local_tensor, world_sz):
    """All-gather tensors that may differ in first dimension across ranks."""
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


def main():
    parser = argparse.ArgumentParser(description="Prompt ensemble test (DDP)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt_embeddings", type=str, default="prompt_embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Init DDP if launched with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        log(f"[Ensemble] DDP initialized: {world_size()} GPUs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log("[Ensemble] Single-GPU mode")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    log(f"[Ensemble] Loading checkpoint: {args.checkpoint}")
    model = BioVidOnlineClassifier.load_from_checkpoint(
        args.checkpoint, map_location="cpu",
        prompt_embeddings_path=args.prompt_embeddings,
    )
    model = model.to(device)
    model.eval()

    num_prompts = model._prompt_embeddings
    num_classes = model.hparams.get("num_classes", 5)
    log(f"[Ensemble] {num_prompts} prompts, {num_classes} classes")

    # Build test dataset with optional distributed sampler
    test_ds = BioVidOnlineDataset(
        split="test",
        labels_csv=data_cfg["labels_csv"],
        frames_root=data_cfg["frames_root"],
        num_classes=data_cfg.get("num_classes", 5),
        resize=data_cfg.get("resize", 128),
        max_frames=data_cfg.get("max_frames", 129),
        sample_rate=data_cfg.get("sample_rate", 2),
    )

    sampler = DistributedSampler(test_ds, shuffle=False) if is_dist() else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_biovid_online,
        pin_memory=True,
        drop_last=False,
    )
    log(f"[Ensemble] Test set: {len(test_ds)} samples, "
        f"{len(test_loader)} batches/GPU")

    # Collect predictions: all_probs[prompt_idx] = list of (B, K-1) cpu tensors
    all_probs = {i: [] for i in range(num_prompts)}
    all_labels = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (video, labels) in enumerate(test_loader):
            video = video.to(device)
            all_labels.append(labels)

            for p_idx in range(num_prompts):
                prompt_emb = getattr(model, f"_prompt_emb_{p_idx}").to(
                    device=device, dtype=video.dtype
                )
                prompt_seq_len = model._prompt_seq_lens[p_idx]

                # Temporarily patch model to use this specific prompt
                orig_n = model._prompt_embeddings
                orig_emb0 = getattr(model, "_prompt_emb_0")
                orig_seq_lens = model._prompt_seq_lens

                model._prompt_embeddings = 1
                setattr(model, "_prompt_emb_0", prompt_emb)
                model._prompt_seq_lens = [prompt_seq_len]

                out = model(video)
                pain_logits = out["pain_coral"]
                probs = torch.sigmoid(pain_logits)
                all_probs[p_idx].append(probs.cpu())

                # Restore
                setattr(model, "_prompt_emb_0", orig_emb0)
                model._prompt_embeddings = orig_n
                model._prompt_seq_lens = orig_seq_lens

            elapsed = time.time() - t0
            speed = (batch_idx + 1) / elapsed
            eta = (len(test_loader) - batch_idx - 1) / speed if speed > 0 else 0
            if rank() == 0 and ((batch_idx + 1) % 5 == 0 or batch_idx == 0):
                print(f"  [GPU0] Batch {batch_idx + 1}/{len(test_loader)} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    # Concatenate local results
    local_labels = torch.cat(all_labels)  # (N_local,)
    for p_idx in range(num_prompts):
        all_probs[p_idx] = torch.cat(all_probs[p_idx])  # (N_local, K-1)

    # Gather across GPUs
    ws = world_size()
    if ws > 1:
        log("[Ensemble] Gathering predictions across GPUs...")
        local_labels = local_labels.to(device)
        global_labels = gather_variable_tensors(local_labels, ws).cpu()
        global_probs = {}
        for p_idx in range(num_prompts):
            local_p = all_probs[p_idx].to(device)
            global_probs[p_idx] = gather_variable_tensors(local_p, ws).cpu()
        if is_dist():
            dist.barrier()
    else:
        global_labels = local_labels
        global_probs = all_probs

    # Only rank 0 computes and prints metrics
    if rank() != 0:
        if is_dist():
            dist.destroy_process_group()
        return

    gt_labels = global_labels.numpy()
    N = len(gt_labels)
    log(f"\n[Ensemble] Collected: {N} samples x {num_prompts} prompts "
        f"({time.time() - t0:.0f}s total)")

    # ---- Single prompt baselines ----
    print("\n" + "=" * 70)
    print("SINGLE PROMPT RESULTS")
    print("=" * 70)
    for p_idx in range(num_prompts):
        preds = coral_prob_to_label(global_probs[p_idx], num_classes).numpy()
        qwk = cohen_kappa_score(gt_labels, preds, weights="quadratic")
        f1 = f1_score(gt_labels, preds, average="macro")
        acc = accuracy_score(gt_labels, preds)
        recall = recall_score(gt_labels, preds, average=None,
                              labels=list(range(num_classes)), zero_division=0)
        recall_str = " | ".join(f"c{i}={recall[i]:.3f}" for i in range(num_classes))
        print(f"  Prompt {p_idx:2d}: QWK={qwk:.4f}  F1={f1:.4f}  Acc={acc:.4f}  [{recall_str}]")

    # ---- Soft voting ----
    print("\n" + "=" * 70)
    print("SOFT VOTING (average CORAL probabilities)")
    print("=" * 70)
    avg_probs = torch.stack([global_probs[i] for i in range(num_prompts)]).mean(dim=0)
    soft_preds = coral_prob_to_label(avg_probs, num_classes).numpy()
    soft_qwk = cohen_kappa_score(gt_labels, soft_preds, weights="quadratic")
    soft_f1 = f1_score(gt_labels, soft_preds, average="macro")
    soft_acc = accuracy_score(gt_labels, soft_preds)
    soft_recall = recall_score(gt_labels, soft_preds, average=None,
                               labels=list(range(num_classes)), zero_division=0)
    print(f"  QWK  = {soft_qwk:.4f}")
    print(f"  F1   = {soft_f1:.4f}")
    print(f"  Acc  = {soft_acc:.4f}")
    print(f"  Per-class recall: " +
          " | ".join(f"c{i}={soft_recall[i]:.3f}" for i in range(num_classes)))

    # ---- Hard voting ----
    print("\n" + "=" * 70)
    print("HARD VOTING (majority vote)")
    print("=" * 70)
    per_prompt_preds = torch.stack([
        coral_prob_to_label(global_probs[i], num_classes) for i in range(num_prompts)
    ])
    hard_preds = []
    for j in range(N):
        votes = per_prompt_preds[:, j].tolist()
        winner = Counter(votes).most_common(1)[0][0]
        hard_preds.append(winner)
    hard_preds = np.array(hard_preds)

    hard_qwk = cohen_kappa_score(gt_labels, hard_preds, weights="quadratic")
    hard_f1 = f1_score(gt_labels, hard_preds, average="macro")
    hard_acc = accuracy_score(gt_labels, hard_preds)
    hard_recall = recall_score(gt_labels, hard_preds, average=None,
                               labels=list(range(num_classes)), zero_division=0)
    print(f"  QWK  = {hard_qwk:.4f}")
    print(f"  F1   = {hard_f1:.4f}")
    print(f"  Acc  = {hard_acc:.4f}")
    print(f"  Per-class recall: " +
          " | ".join(f"c{i}={hard_recall[i]:.3f}" for i in range(num_classes)))

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    p0_preds = coral_prob_to_label(global_probs[0], num_classes).numpy()
    p0_qwk = cohen_kappa_score(gt_labels, p0_preds, weights="quadratic")
    print(f"  Single prompt[0]:  QWK = {p0_qwk:.4f}")
    print(f"  Soft voting:       QWK = {soft_qwk:.4f}  (delta = {soft_qwk - p0_qwk:+.4f})")
    print(f"  Hard voting:       QWK = {hard_qwk:.4f}  (delta = {hard_qwk - p0_qwk:+.4f})")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
