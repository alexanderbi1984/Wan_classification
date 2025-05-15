# Feature Extraction in Wan: VAE and xDiT

This document summarizes how the VAE and xDiT (transformer) models are instantiated and used in the Wan codebase, and how to extract their features for downstream tasks.

---

## 1. VAE (Variational Autoencoder)

- **Location:** `wan/modules/vae.py`
- **Class:** `WanVAE`
- **Instantiation Example:**
  ```python
  self.vae = WanVAE(
      vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
      device=self.device)
  ```
- **Feature Extraction:**
  - Use the `encode` method to obtain VAE latents for a video or image sequence.
  - Example:
    ```python
    latents = self.vae.encode([video_tensor])[0]
    # video_tensor: shape [C, T, H, W]
    ```
  - The output is the latent representation of the input, suitable for use as a feature vector.

- **Batch Extraction Script:**
  - A ready-to-use script is provided at `feature_extraction/extract_vae_latents.py` to extract VAE latents from a single video file or from all `.mp4` videos in a folder.

  - **Single video usage:**
    ```sh
    python feature_extraction/extract_vae_latents.py \
        --video path/to/input_video.mp4 \
        --vae_ckpt path/to/vae_checkpoint.pth \
        --output output_latents.npy
    ```
  - **Batch processing all mp4 files in a folder:**
    ```sh
    python feature_extraction/extract_vae_latents.py \
        --video path/to/input_folder \
        --vae_ckpt path/to/vae_checkpoint.pth \
        --output path/to/output_folder
    ```
    - Output files will use the same base name as the input `.mp4` but with a `.npy` extension (e.g., `video1.mp4` → `video1.npy`).

---

## 2. xDiT (Transformer Backbone)

- **Location:** `wan/modules/model.py`
- **Class:** `WanModel`
- **Instantiation Example:**
  ```python
  self.model = WanModel.from_pretrained(checkpoint_dir)
  ```
- **Feature Extraction:**
  - The model processes VAE latents and outputs denoised latents during the diffusion process.
  - You can extract two types of features:
    - **Patchified transformer features** (sequence of patch tokens, shape `[seq_len, hidden_dim]`)
    - **Unpatchified denoised latents** (final output, shape `[C, F, H, W]`)
  - A ready-to-use script is provided at `feature_extraction/extract_xdit_features.py` to extract xDiT features from a single video file or from all videos in a folder.

  - **Single video usage (patchified features):**
    ```sh
    python feature_extraction/extract_xdit_features.py \
        --video path/to/input_video.mp4 \
        --checkpoint_dir path/to/model_checkpoint_directory \
        --output output_xdit_features.npy \
        --device cuda
    ```
    - Output: Patchified transformer features, shape `[seq_len, hidden_dim]` (e.g., `[320, 1536]`).

  - **Single video usage (unpatchified/denoised latents):**
    ```sh
    python feature_extraction/extract_xdit_features.py \
        --video path/to/input_video.mp4 \
        --checkpoint_dir path/to/model_checkpoint_directory \
        --output output_xdit_latents.npy \
        --device cuda \
        --unpatchify_output
    ```
    - Output: Denoised latent, shape `[C, F, H, W]` (e.g., `[16, 5, 16, 16]`), directly comparable to VAE latents.

  - **Single video usage (temporal features, CTHW):**
    ```sh
    python feature_extraction/extract_xdit_features.py \
        --video path/to/input_video.mp4 \
        --checkpoint_dir path/to/model_checkpoint_directory \
        --output output_temporal_features.npy \
        --device cuda \
        --return_temporal_features
    ```
    - Output: Temporal features, shape `[C, T, H, W]` (e.g., `[1536, 5, 16, 16]`).
    - Note: The output is permuted from the model's native `[T, H, W, C]` to `[C, T, H, W]` for PyTorch compatibility.

  - **Batch processing all videos in a folder:**
    ```sh
    python feature_extraction/extract_xdit_features.py \
        --video path/to/input_folder \
        --checkpoint_dir path/to/model_checkpoint_directory \
        --output path/to/output_folder \
        --device cuda [--unpatchify_output]
    ```
    - Output files will use the same base name as the input video but with a `.npy` extension.

  - **Options:**
    - `--unpatchify_output`: If set, saves the final denoised latent `[C, F, H, W]` (recommended for downstream tasks that expect VAE-like features).
    - If not set, saves the patchified transformer features `[seq_len, hidden_dim]`.
    - `--feature_layer`: You can specify which transformer block's output to extract (default: final block).
    - `--use_text_context`: Use actual T5 encoder for context (default: dummy context for efficiency).

  - **Output shapes:**
    - Patchified: `[seq_len, hidden_dim]` (e.g., `[320, 1536]`)
    - Unpatchified: `[C, F, H, W]` (e.g., `[16, 5, 16, 16]`)

---

## 3. Usage in Pipelines

- **Pipelines:**
  - `WanI2V`, `WanT2V`, and `WanFLF2V` (in `image2video.py`, `text2video.py`, `first_last_frame2video.py`)
  - These classes instantiate both `WanVAE` and `WanModel` and use them in their `generate` methods.

- **Current Feature Access:**
  - VAE latents can be accessed directly via the `encode` method or via the provided batch script.
  - xDiT features can be accessed via the provided script, with options for patchified or unpatchified output.

---

## 4. Next Steps

- For VAE latents: Use the existing `encode` method or batch script.
- For xDiT features: Use the provided script with the desired output option (`--unpatchify_output` for denoised latents).

---

*This document serves as a reference for implementing and reviewing feature extraction from the Wan VAE and xDiT models, including both patchified transformer features and denoised latent outputs suitable for downstream tasks.* 