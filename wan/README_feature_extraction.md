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
  - To extract intermediate features (e.g., patch embeddings, block outputs, or final transformer output), you need to:
    - Modify the `WanModel` class to return or expose the desired internal representations.
    - This may involve editing the `forward` method to return features from specific layers or blocks.
  - Example (conceptual):
    ```python
    # After modification
    features, output = self.model(..., return_features=True)
    ```

---

## 3. Usage in Pipelines

- **Pipelines:**
  - `WanI2V`, `WanT2V`, and `WanFLF2V` (in `image2video.py`, `text2video.py`, `first_last_frame2video.py`)
  - These classes instantiate both `WanVAE` and `WanModel` and use them in their `generate` methods.

- **Current Feature Access:**
  - VAE latents can be accessed directly via the `encode` method or via the provided batch script.
  - xDiT features require code modification to expose intermediate outputs.

---

## 4. Next Steps

- For VAE latents: Use the existing `encode` method or batch script.
- For xDiT features: Modify `WanModel` to return the desired features.

---

*This document serves as a reference for implementing and reviewing feature extraction from the Wan VAE and xDiT models.* 