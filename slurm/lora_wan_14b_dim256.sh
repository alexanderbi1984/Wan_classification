#!/bin/bash -l
#SBATCH --job-name=WanLoRA14B_dim256
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,wlan"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512

cd ~/Wan_classification

echo "[$(date +%T)] Experiment: feature_dim=256, fusion_dim=256 (larger downstream head)"
echo "[$(date +%T)] Config: config_pain/config_lora_dim256.yaml"

srun --ntasks=1 --gpus-per-node=${SLURM_GPUS_PER_NODE} \
  torchrun --standalone --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  train_online_wan.py \
    --config config_pain/config_lora_dim256.yaml \
    --n_gpus ${SLURM_GPUS_PER_NODE}

echo "[$(date +%T)] Done."
echo "TensorBoard: tensorboard --logdir results/"
