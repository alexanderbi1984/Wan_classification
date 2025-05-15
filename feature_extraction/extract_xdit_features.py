# feature_extraction/extract_xdit_features.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import cv2
from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
import math
from tqdm import tqdm

class FeatureExtractorWanModel(torch.nn.Module):
    """Wrapper for WanModel that extracts intermediate features during forward pass."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        return_features=False,
        return_temporal_features=False,
        feature_layer=-1,  # -1 means final output, otherwise specific block index
    ):
        """Modified forward method to return intermediate features.
        
        Args:
            x: Input tensor
            t: Timestep
            context: Context tensor
            seq_len: Sequence length
            clip_fea: CLIP features (optional)
            y: Optional conditioning
            return_features: Whether to return intermediate features
            feature_layer: Which layer to extract features from (-1 for final output)
            
        Returns:
            Tuple of (output, features) if return_features=True, otherwise just output
        """
        model = self.model
        
        # Print shape information for debugging
        print(f"Input shapes: x={[x_i.shape for x_i in x]}, t={t.shape}")
        
        if len(x) != 1:
            print(f"WARNING: Expected x to be a list with 1 element, but got {len(x)} elements")
        
        # Temporary fix for unpacking error - handle different input shapes
        try:
            b, c, f, h, w = x[0].shape
        except ValueError as e:
            print(f"NO BATCH DIMENSION: {e}")
            print(f"x[0] shape: {x[0].shape}")
            # Try to adapt to different input shapes
            if len(x[0].shape) == 4:  # Handle case where batch dimension is missing
                c, f, h, w = x[0].shape
                b = 1
                x[0] = x[0].unsqueeze(0)  # Add batch dimension
                print(f"Adapted shape to: {x[0].shape}")
            else:
                raise
        
        # Patchify using the model's patch_embedding layer (as in WanModel.forward)
        x = [model.patch_embedding(x_i) if x_i.dim() == 5 else model.patch_embedding(x_i.unsqueeze(0)) for x_i in x]
        grid_sizes = torch.stack(
            [torch.tensor(x_i.shape[2:], dtype=torch.long) for x_i in x])
        x = [x_i.flatten(2).transpose(1, 2) for x_i in x]
        seq_lens = torch.tensor([x_i.size(1) for x_i in x], dtype=torch.long)
        x = torch.cat([
            torch.cat([x_i, x_i.new_zeros(1, seq_len - x_i.size(1), x_i.size(2))],
                      dim=1) for x_i in x
        ])
        
        # prepare context
        context_lens = None
        if context is not None:
            if isinstance(context, list):
                context, context_lens = context
                
        # prepare time embedding
        sin_emb = sinusoidal_embedding_1d(model.freq_dim, t.to(x.dtype))
        e = model.time_embedding(sin_emb.float())
        
        # prepare rope freqs
        freqs = model.freqs.to(x.device, dtype=torch.complex128)
        
        # apply blocks
        block_features = []  # Store features from each block
        
        for i, block in enumerate(model.blocks):
            x = block(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
            if return_features and (feature_layer == i or feature_layer == -1 and i == len(model.blocks) - 1):
                block_features.append(x.clone())
        
        # Extract temporal features
        temporal_features = self.extract_temporal_features(x, grid_sizes)
        # add sanity check
        assert temporal_features.shape[0] == 1, "Expected temporal features to be [B, T, H, W, dim]"
        temporal_features = temporal_features[0]

        
        # head
        x = model.head(x, e)
        
        # unpatchify
        x = model.unpatchify(x, grid_sizes)
        
        if return_features:
            print(f"RETURNING FEATURES: {x.shape}, {block_features[0].shape}, {grid_sizes.shape}")
            return x, block_features, grid_sizes
        elif return_temporal_features:
            print(f"RETURNING TEMPORAL FEATURES: {temporal_features.shape}")
            return x, block_features, grid_sizes, temporal_features
        else:
            return x

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def extract_temporal_features(self, features, grid_sizes):
        """
        Reshape [B, L, dim] token features to [B, T, H, W, dim] based on grid size.
        
        Args:
            features (Tensor): Transformer output of shape [B, L, dim]
            grid_sizes (Tensor): Grid sizes used for patch embedding, shape [B, 3] with (T', H', W')
        
        Returns:
            Tensor: Reshaped features with temporal dimension [B, T, H, W, dim]
        """
        B, L, D = features.shape
        assert grid_sizes.shape[1] == 3, "Expected grid_sizes to be [B, 3] (T', H', W')"
        temporal_features = []

        for b in range(B):
            T, H, W = grid_sizes[b].tolist()
            expected_L = T * H * W
            assert features[b].shape[0] == expected_L, f"Expected L={expected_L}, got {features[b].shape[0]}"
            feat = features[b].view(T, H, W, D)  # [T, H, W, D]
            temporal_features.append(feat)

        temporal_features = torch.stack(temporal_features, dim=0)  # [B, T, H, W, D]
        return temporal_features


def read_video_frames(video_path, max_frames=129, resize=128, sample_fps=3):
    """
    Read and sample frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to extract.
        resize (int): Resize frames to (resize, resize).
        sample_fps (int): Target sampling frame rate (e.g., 3 FPS).
        
    Returns:
        List[Tensor]: List of frames as torch tensors (C, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = max(int(input_fps / sample_fps), 1)
    
    frames, count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            frame = torch.from_numpy(frame).float() / 255.0  # (H, W, C)
            frame = frame.permute(2, 0, 1)  # (C, H, W)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        count += 1
    cap.release()
    return frames


def process_and_save_features(video_path, vae, model, text_encoder, output_path, args):
    """Process a video and save extracted features."""
    frames = read_video_frames(video_path, max_frames=args.max_frames, resize=args.resize)
    if not frames:
        print(f"No frames extracted from {video_path}.")
        return
    
    video_tensor = torch.stack(frames, dim=1).to(args.device)  # (C, T, H, W)
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Get VAE latents
    with torch.no_grad():
        latents = vae.encode([video_tensor])
        print(f"VAE latents shape: {[l.shape for l in latents]}")
    
    # Prepare model inputs
    target_shape = latents[0].shape
    # Get patch size from the model
    patch_size = model.model.patch_size
    print(f"Patch size: {patch_size}")
    
    # Extract dimensions safely
    if len(target_shape) == 5:  # [B, C, F, H, W]
        _, _, F, H, W = target_shape
    elif len(target_shape) == 4:  # [C, F, H, W]
        _, F, H, W = target_shape
    else:
        raise ValueError(f"Unexpected latent shape: {target_shape}")
    
    print(f"Extracted dimensions: F={F}, H={H}, W={W}")
    
    seq_len = math.ceil((H * W) / (patch_size[1] * patch_size[2]) * F)
    print(f"Calculated seq_len: {seq_len}")
    
    # Dynamically detect context dimension
    if hasattr(model.model, 'dim'):
        context_dim = model.model.dim
    elif hasattr(model.model, 'config') and hasattr(model.model.config, 'dim'):
        context_dim = model.model.config.dim
    else:
        # Default fallback dimension - typical for T5 encoder output
        context_dim = 768
        print(f"Warning: Could not detect context dimension, using default: {context_dim}")
    
    print(f"Using context dimension: {context_dim}")
    
    # Get text context (using empty string as we just need the structure)
    if args.use_text_context:
        # Process on CPU, then move only the results to GPU
        context = text_encoder([""], torch.device('cpu'))
        context = [t.to(args.device) for t in context]
    else:
        # Use dummy context with detected dimension
        dummy_context = torch.zeros(1, 1, context_dim).to(args.device)
        context = [dummy_context, torch.tensor([1]).to(args.device)]
    
    # Set a fixed timestep for feature extraction (e.g., t=0 for denoised features)
    t = torch.tensor([0], device=args.device)
    
    # Extract features
    with torch.no_grad():
        try:
            print("Calling model.forward to extract features...")
            if args.return_temporal_features:
                out, features, grid_sizes, temporal_features = model(
                    latents, 
                    t=t, 
                    context=context, 
                    seq_len=seq_len,
                    return_features=False,
                    return_temporal_features=True,
                    feature_layer=args.feature_layer
                )
            else:
                out, features, grid_sizes = model(
                    latents, 
                    t=t, 
                    context=context, 
                    seq_len=seq_len,
                    return_features=True,
                    return_temporal_features=False,
                    feature_layer=args.feature_layer
                )
            
            # Sanity check for output shapes
            if features is None or len(features) == 0 and not args.return_temporal_features:
                raise ValueError("Model returned empty features")
            elif temporal_features is None or len(temporal_features) == 0 and args.return_temporal_features:
                raise ValueError("Model returned empty temporal features")
            
            # Check feature shapes (first dimension should match batch size)
            batch_size = 1  # Assume batch size 1 for single video
            for i, feat in enumerate(features):
                print(f"Feature {i} shape: {feat.shape}")
                if feat.shape[0] != batch_size and args.verbose:
                    print(f"Warning: Feature {i} has batch dimension {feat.shape[0]}, expected {batch_size}")
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Save features
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.unpatchify_output:
        # Save the final model output (already in [C, F, H, W] format)
        if isinstance(out, list) and len(out) == 1:
            out_to_save = out[0]
        else:
            out_to_save = out
        np.save(output_path, out_to_save.cpu().numpy())
        print(f"Saved unpatchified xDiT features to {output_path}, shape: {out_to_save.shape}")
    elif args.return_temporal_features:
        # Save temporal features in (C, T, H, W) format instead of (T, H, W, C)
        # temporal_features is (T, H, W, C), so we need to permute axes
        temporal_features_cthw = temporal_features.permute(3, 0, 1, 2).contiguous()
        np.save(output_path, temporal_features_cthw.cpu().numpy())
        print(f"Saved temporal features to {output_path}, shape: {temporal_features_cthw.shape}")
    else:
        features_np = [f.squeeze(0).cpu().numpy() if f.shape[0] == 1 else f.cpu().numpy() for f in features]
        np.save(output_path, features_np)
        print(f"Saved xDiT features to {output_path}, shapes: {[f.shape for f in features_np]}")


def main():
    try:
        parser = argparse.ArgumentParser(description="Extract xDiT features from a video or all videos in a folder.")
        parser.add_argument('--video', type=str, required=True, help='Path to the input video file or folder.')
        parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the model checkpoint directory.')
        parser.add_argument('--output', type=str, required=True, help='Path to save the extracted features (.npy) or directory for multiple files.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu).')
        parser.add_argument('--resize', type=int, choices=[128, 192, 256, 480, 720], default=128,
                         help='Resize frames to (height, width) = (R, R). Choose from 128, 192, 256, 480, 720.')
        parser.add_argument('--max_frames', type=int, default=129, help='Maximum number of frames to process (max 129).')
        parser.add_argument('--feature_layer', type=int, default=-1, 
                         help='Which layer to extract features from (-1 for final output, otherwise specific block index).')
        parser.add_argument('--model_type', type=str, default='t2v', choices=['t2v', 'i2v', 'flf2v'],
                         help='Type of model to use (t2v, i2v, or flf2v).')
        parser.add_argument('--use_text_context', action='store_true',
                         help='Use actual T5 encoder for context. If False, use a dummy context to save memory.')
        parser.add_argument('--verbose', action='store_true',
                         help='Enable verbose output including feature shapes.')
        parser.add_argument('--unpatchify_output', action='store_true',
                          help='If set, unpatchify the features before saving (output shape [C, F, H, W]).')
        parser.add_argument('--return_temporal_features', action='store_true',
                          help='If set, return temporal features.')
        args = parser.parse_args()
        
        # Load models
        device = torch.device(args.device)
        
        # Load VAE
        vae_checkpoint = os.path.join(args.checkpoint_dir, "Wan2.1_VAE.pth")  # Updated path based on actual file
        print(f"Loading VAE from: {vae_checkpoint}")
        vae = WanVAE(vae_pth=vae_checkpoint, device=device)
        
        # Load T5 text encoder only if needed
        text_encoder = None
        if args.use_text_context:
            t5_checkpoint = os.path.join(args.checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")  # Updated path
            t5_tokenizer = os.path.join(args.checkpoint_dir, "google/umt5-xxl")  # Updated path
            print(f"Loading T5 encoder from: {t5_checkpoint}")
            print(f"Loading T5 tokenizer from: {t5_tokenizer}")
            text_encoder = T5EncoderModel(
                text_len=512,  # Default value
                dtype=torch.float16,
                device=torch.device('cpu'),  # Keep on CPU to save VRAM
                checkpoint_path=t5_checkpoint,
                tokenizer_path=t5_tokenizer
            )
        
        # Load WanModel and wrap it with our feature extractor
        print(f"Loading WanModel from: {args.checkpoint_dir}")
        original_model = WanModel.from_pretrained(args.checkpoint_dir)
        original_model.eval().requires_grad_(False)
        original_model.to(device)
        
        # Create our feature extractor wrapper - properly wrapping the original model
        model = FeatureExtractorWanModel(original_model)
        
        # Process videos
        if os.path.isdir(args.video):
            video_dir = args.video
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            mp4_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
            if not mp4_files:
                print(f"No mp4 files found in {video_dir}.")
                return
            for fname in tqdm(mp4_files):
                video_path = os.path.join(video_dir, fname)
                base = os.path.splitext(fname)[0]
                output_path = os.path.join(output_dir, base + '.npy')
                process_and_save_features(video_path, vae, model, text_encoder, output_path, args)
        else:
            # single file: decide output file name
            if os.path.isdir(args.output):
                base = os.path.splitext(os.path.basename(args.video))[0]
                output_path = os.path.join(args.output, base + '.npy')
            else:
                output_path = args.output
            process_and_save_features(args.video, vae, model, text_encoder, output_path, args)
    except Exception as e:
        import traceback
        print(f"Error in main function: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()