#!/bin/bash
#SBATCH --job-name=GradCAM
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/GradCAM_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=config_pain/config_lora_t100_aug3.yaml
CKPT=results/wan_lora_14B_biovid_t100_aug3_20260305_002030/checkpoints/wan_lora_14B_biovid_t100_aug3_best-epoch=14-val_pain_QWK=0.430.ckpt
OUT_DIR=results/gradcam

python scripts/visualize_gradcam.py \
    --config ${CONFIG} \
    --checkpoint ${CKPT} \
    --output_dir ${OUT_DIR} \
    --n_per_class 10 \
    --n_display_frames 8 \
    --split test \
    --device cuda

echo "[$(date)] GradCAM visualization complete"
