#!/bin/bash -l
#SBATCH --job-name=DiagnoseGap
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nbi1@binghamton.edu

set -euo pipefail

module load miniconda
eval "$(/opt/ohpc/pub/apps/miniforge3/bin/conda shell.bash hook)"
conda activate wan || { echo "conda activate wan failed"; exit 1; }

mkdir -p logs
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
export MASTER_PORT=29525

cd ~/Wan_classification

CHECKPOINT="results/wan_lora_14B_biovid_t100_aug2_20260303_052322/checkpoints/wan_lora_14B_biovid_t100_aug2_best-epoch=21-val_pain_QWK=0.414.ckpt"

echo "[$(date +%T)] Diagnosing val-test gap with best t100_aug2 checkpoint"
echo "[$(date +%T)] Checkpoint: ${CHECKPOINT}"

srun --ntasks=1 --gpus-per-node=${SLURM_GPUS_PER_NODE} \
  torchrun --standalone --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  scripts/diagnose_val_test_gap.py \
    --config config_pain/config_lora_t100_aug2.yaml \
    --checkpoint "${CHECKPOINT}" \
    --batch_size 4 \
    --num_workers 4

echo "[$(date +%T)] Done."
