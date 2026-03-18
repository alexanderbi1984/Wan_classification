#!/bin/bash
#SBATCH --job-name=DaC
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/DaC_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CKPT_DIR=Wan2.1-T2V-14B
CONFIG=config_pain/config_lora_t100_aug3.yaml
OUT_DIR=results/dac

python scripts/diffusion_classifier.py \
    --checkpoint_dir ${CKPT_DIR} \
    --config ${CONFIG} \
    --output_dir ${OUT_DIR} \
    --timesteps 200 500 800 \
    --n_noise_samples 1 \
    --split test \
    --device cuda

echo "[$(date)] DaC evaluation complete"
