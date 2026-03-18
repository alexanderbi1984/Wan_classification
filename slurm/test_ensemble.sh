#!/bin/bash -l
#SBATCH --job-name=WanLoRA14B_Ensemble
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nbi1@binghamton.edu

# Sequential ensemble: loads one model at a time -> only 1 GPU needed
# Works on any GPU with >= 40 GB VRAM (A100-40GB, A100-80GB, H100, etc.)
# For smaller GPUs, reduce --batch_size (e.g., 4 or 8)

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

# Find best checkpoints from top-3 CORAL experiments
CKPT1=$(find results/ -path "*t100_aug3_20*" -name "*best*" -name "*.ckpt" | grep -v "mixup\|mae\|rank" | head -1)
CKPT2=$(find results/ -path "*t100_aug2_20*" -name "*best*" -name "*.ckpt" | head -1)
CKPT3=$(find results/ -path "*dim256_aug2_20*" -name "*best*" -name "*.ckpt" | head -1)

echo "[$(date +%T)] Sequential Ensemble of 3 models (single GPU):"
echo "  1: $CKPT1"
echo "  2: $CKPT2"
echo "  3: $CKPT3"

for C in "$CKPT1" "$CKPT2" "$CKPT3"; do
    if [ -z "$C" ]; then
        echo "ERROR: Missing checkpoint"
        exit 1
    fi
done

python scripts/test_ensemble.py \
    --config config_pain/config_lora_t100_aug3.yaml \
    --ckpts "$CKPT1" "$CKPT2" "$CKPT3" \
    --batch_size 24

echo "[$(date +%T)] Done."
