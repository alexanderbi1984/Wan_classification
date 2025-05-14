import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def validate_xdit_features(feature_path, visualize=False, num_channels=4, num_tokens=4):
    if not os.path.exists(feature_path):
        print(f"[ERROR] File not found: {feature_path}")
        return

    features = np.load(feature_path, allow_pickle=True)
    print(f"[INFO] Features loaded from: {feature_path}")
    print(f"[INFO] Type: {type(features)}")
    if isinstance(features, list):
        print(f"[INFO] List of {len(features)} arrays.")
        features = features[0]  # Assume first is main feature for single-sample case
    print(f"[INFO] Shape: {features.shape}")
    print(f"[INFO] Dtype: {features.dtype}")
    print(f"[INFO] Memory: {features.nbytes / (1024 * 1024):.2f} MB")

    print(f"[INFO] Min: {features.min():.4f}")
    print(f"[INFO] Max: {features.max():.4f}")
    print(f"[INFO] Mean: {features.mean():.4f}")
    print(f"[INFO] Std Dev: {features.std():.4f}")

    if np.isnan(features).any():
        print("[WARNING] Features contain NaNs!")

    # Print first few values for inspection
    print("[INFO] First few values:")
    print(features.flatten()[:10])

    # Visualize a few feature channels/tokens
    if visualize:
        C = features.shape[-1]
        T = features.shape[0]
        for i in range(min(num_tokens, T)):
            for j in range(min(num_channels, C)):
                plt.plot(features[i, :, j]) if features.ndim == 3 else plt.plot(features[i, j])
                plt.title(f'Feature Token {i}, Channel {j}')
                plt.xlabel('Feature Index')
                plt.ylabel('Value')
                plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a .npy xDiT feature file.")
    parser.add_argument('--feature_path', type=str, required=True, help='Path to .npy feature file.')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize a few feature channels/tokens.')
    parser.add_argument('--num_channels', type=int, default=4, help='Number of channels to visualize.')
    parser.add_argument('--num_tokens', type=int, default=4, help='Number of tokens to visualize.')
    args = parser.parse_args()

    validate_xdit_features(
        feature_path=args.feature_path,
        visualize=args.visualize,
        num_channels=args.num_channels,
        num_tokens=args.num_tokens
    ) 