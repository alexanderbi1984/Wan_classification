import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import cv2
from wan.modules.vae import WanVAE


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


def process_and_save_latents(video_path, vae, output_path, args):
    frames = read_video_frames(video_path, max_frames=args.max_frames, resize=args.resize)
    if not frames:
        print(f"No frames extracted from {video_path}.")
        return
    video_tensor = torch.stack(frames, dim=1).to(args.device)  # (C, T, H, W)
    with torch.no_grad():
        latents = vae.encode([video_tensor])[0].cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, latents)
    print(f"Saved VAE latents to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract VAE latents from a video or all videos in a folder using WanVAE.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file or folder.')
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to the VAE checkpoint file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the extracted latents (.npy) or directory for multiple files.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--resize', type=int, choices=[128, 192, 256, 480, 720], default=128,
                        help='Resize frames to (height, width) = (R, R). Choose from 128, 192, 256, 480, 720.')
    parser.add_argument('--max_frames', type=int, default=129, help='Maximum number of frames to process (max 129).')
    args = parser.parse_args()

    vae = WanVAE(vae_pth=args.vae_ckpt, device=args.device)

    if os.path.isdir(args.video):
        video_dir = args.video
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        mp4_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
        if not mp4_files:
            print(f"No mp4 files found in {video_dir}.")
            return
        for fname in mp4_files:
            video_path = os.path.join(video_dir, fname)
            base = os.path.splitext(fname)[0]
            output_path = os.path.join(output_dir, base + '.npy')
            process_and_save_latents(video_path, vae, output_path, args)
    else:
        # single file: decide output file name
        if os.path.isdir(args.output):
            base = os.path.splitext(os.path.basename(args.video))[0]
            output_path = os.path.join(args.output, base + '.npy')
        else:
            output_path = args.output
        process_and_save_latents(args.video, vae, output_path, args)


if __name__ == '__main__':
    main()