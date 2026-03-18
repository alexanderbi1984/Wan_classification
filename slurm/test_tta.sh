#!/bin/bash -l
#SBATCH --job-name=WanLoRA14B_TTA
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nbi1@binghamton.edu

# TTA: single model inference with augmentation -> only 1 GPU needed
# Works on any GPU with >= 40 GB VRAM

set -euo pipefail

module load miniconda
eval "$(/opt/ohpc/pub/apps/miniforge3/bin/conda shell.bash hook)"
conda activate wan || { echo "conda activate wan failed"; exit 1; }

mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/Wan_classification

# Find best checkpoint from t100_aug3 experiment
CKPT=$(find results/ -path "*t100_aug3_20*" -name "*best*" -name "*.ckpt" | grep -v "mixup\|mae\|rank" | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: Could not find t100_aug3 best checkpoint"
    exit 1
fi

echo "[$(date +%T)] TTA Test with n_aug=10 (single GPU)"
echo "[$(date +%T)] Checkpoint: $CKPT"

python scripts/test_tta.py \
    --config config_pain/config_lora_t100_aug3.yaml \
    --ckpt_path "$CKPT" \
    --n_aug 10 \
    --n_gpus 1 \
    --batch_size 24

echo "[$(date +%T)] Done."
