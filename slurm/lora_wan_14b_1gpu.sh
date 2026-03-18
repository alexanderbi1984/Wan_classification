#!/bin/bash -l
#SBATCH --job-name=WanLoRA1G
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nbi1@binghamton.edu

set -euo pipefail

module load miniconda
eval "$(/opt/ohpc/pub/apps/miniforge3/bin/conda shell.bash hook)"
conda activate wan || { echo "conda activate wan failed"; exit 1; }

mkdir -p logs results
echo "[$(date +%T)] NodeList=$SLURM_NODELIST  NNODES=$SLURM_NNODES"
echo "[$(date +%T)] GPUS_PER_NODE=$SLURM_GPUS_PER_NODE  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
timeout 5 nvidia-smi -L || echo "[warn] nvidia-smi -L timeout, continuing..."

# ---------- Environment ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/Wan_classification

echo "[$(date +%T)] Wan 14B DiT LoRA fine-tuning — SINGLE GPU mode"
echo "[$(date +%T)] Config: config_pain/config_lora.yaml"
echo "[$(date +%T)] batch_size=1, accumulate_grad_batches=8, grad_checkpointing=ON"
echo "[$(date +%T)] Estimated VRAM: ~22 GB (bf16 + grad ckpt)"

python train_online_wan.py \
  --config config_pain/config_lora.yaml \
  --n_gpus 1

echo "[$(date +%T)] Done."
echo "TensorBoard: tensorboard --logdir results/"
