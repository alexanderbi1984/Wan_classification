"""Pre-compute T5 embeddings for the prompt pool and save as a .pt file.

Usage (GPU node):
    python scripts/precompute_prompt_embeddings.py \
        --checkpoint_dir Wan2.1-T2V-14B \
        --output prompt_embeddings.pt

The output file contains:
    {
        "prompts": List[str],
        "embeddings": List[Tensor],    # each (seq_len_i, 4096)
        "seq_lens": List[int],
    }

During training, the classifier loads this file and randomly samples
one embedding per batch (training) or uses a fixed one (val/test).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.t5 import T5EncoderModel

PROMPT_POOL = [
    # Core task descriptions
    "A close-up video of a person's face showing pain expression.",
    "A video recording of a patient during a pain assessment test.",
    "Facial reaction of a subject undergoing physical pain stimulation.",
    # Emphasis on facial movement
    "Close-up of facial muscle contractions during painful stimuli.",
    "A person grimacing in response to increasing levels of pain.",
    "Detailed facial expressions of a human subject feeling distressed.",
    # Emphasis on clinical / experimental setting
    "Clinical monitoring of facial responses to thermal pain.",
    "A laboratory video of a person reacting to painful sensations.",
    "High-resolution facial recording during a pain sensitivity experiment.",
    # Emphasis on emotion and intensity
    "The subtle and intense facial manifestations of physical pain.",
    "A subject's facial changes while experiencing various pain intensities.",
    "Capturing the involuntary facial expressions of a person in pain.",
    # Concise descriptions
    "Human facial response to painful stimulation.",
    "Pain-related facial expressions in a controlled experiment.",
    "Recording of a person's face during a painful episode.",
]


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 prompt embeddings")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to Wan model dir containing T5 weights and tokenizer")
    parser.add_argument("--output", type=str, default="prompt_embeddings.pt",
                        help="Output .pt file path")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto' (cuda if available)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint_path = os.path.join(args.checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(args.checkpoint_dir, "google", "umt5-xxl")

    print(f"[T5] Loading model from {checkpoint_path}")
    print(f"[T5] Tokenizer: {tokenizer_path}")
    print(f"[T5] Device: {device}")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )

    print(f"[T5] Encoding {len(PROMPT_POOL)} prompts...")
    with torch.no_grad():
        embeddings = text_encoder(PROMPT_POOL, device=device)

    # Each embedding is (seq_len_i, 4096) in bf16 — convert to float32 for storage
    embeddings_f32 = [e.cpu().float() for e in embeddings]
    seq_lens = [e.shape[0] for e in embeddings_f32]

    result = {
        "prompts": PROMPT_POOL,
        "embeddings": embeddings_f32,
        "seq_lens": seq_lens,
    }

    torch.save(result, args.output)
    print(f"[T5] Saved {len(PROMPT_POOL)} embeddings to {args.output}")
    for i, (p, s) in enumerate(zip(PROMPT_POOL, seq_lens)):
        print(f"  [{i:2d}] tokens={s:3d}  \"{p[:60]}...\"")
    print(f"[T5] Total file size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
