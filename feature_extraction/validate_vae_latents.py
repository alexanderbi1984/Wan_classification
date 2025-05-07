import argparse
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def validate_latents(latent_path, input_frames=None, input_size=None, visualize=False):
    if not os.path.exists(latent_path):
        print(f"[ERROR] File not found: {latent_path}")
        return

    latents = np.load(latent_path)
    print(f"[INFO] Latents loaded from: {latent_path}")
    print(f"[INFO] Shape: {latents.shape}")
    print(f"[INFO] Dtype: {latents.dtype}")
    print(f"[INFO] Memory: {latents.nbytes / (1024 * 1024):.2f} MB")

    print(f"[INFO] Min: {latents.min():.4f}")
    print(f"[INFO] Max: {latents.max():.4f}")
    print(f"[INFO] Mean: {latents.mean():.4f}")
    print(f"[INFO] Std Dev: {latents.std():.4f}")

    if np.isnan(latents).any():
        print("[WARNING] Latents contain NaNs!")

    # Inferred original input resolution and frame count
    estimated_input_frames = latents.shape[1] * 4
    estimated_input_size = latents.shape[2] * 8
    print(f"[INFO] Estimated original input: ~{estimated_input_frames} frames, ~{estimated_input_size}x{estimated_input_size} px")

    # Compare with known input if provided
    if input_frames is not None:
        expected_T = math.ceil(input_frames / 4)
        actual_T = latents.shape[1]
        print(f"[CHECK] Input frames provided: {input_frames}")
        print(f"[CHECK] Expected latent T: ~{expected_T}, Got: {actual_T}")
    
    if input_size is not None:
        expected_HW = input_size // 8
        actual_H, actual_W = latents.shape[2], latents.shape[3]
        print(f"[CHECK] Input size provided: {input_size}x{input_size}")
        print(f"[CHECK] Expected latent H/W: {expected_HW}, Got: {actual_H}/{actual_W}")

    # Visualize slices
    if visualize:
        channels = min(4, latents.shape[0])
        for i in range(channels):
            plt.imshow(latents[i, 0], cmap='viridis')
            plt.title(f'Latent Channel {i}, Frame 0')
            plt.colorbar()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a .npy latent file generated from WanVAE.")
    parser.add_argument('--latent_path', type=str, required=True, help='Path to .npy latent file.')
    parser.add_argument('--input_frames', type=int, help='Original number of frames in the input video (optional).')
    parser.add_argument('--input_size', type=int, help='Original frame resolution (optional, e.g., 128, 256).')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize a few latent slices.')
    args = parser.parse_args()

    validate_latents(
        latent_path=args.latent_path,
        input_frames=args.input_frames,
        input_size=args.input_size,
        visualize=args.visualize
    )