#!/bin/bash -l
#SBATCH --job-name=PromptEnsemble_test
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
export MASTER_PORT=29520

cd ~/Wan_classification

CKPT="results/wan_lora_14B_biovid_dim256_prompt_20260303_095627/checkpoints/wan_lora_14B_biovid_dim256_prompt_best-epoch=14-val_pain_QWK=0.401.ckpt"

echo "[$(date +%T)] Running prompt ensemble test (4-GPU DDP)"
echo "[$(date +%T)] Checkpoint: ${CKPT}"

torchrun --standalone --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  scripts/test_prompt_ensemble.py \
    --config config_pain/config_lora_dim256_prompt.yaml \
    --checkpoint "${CKPT}" \
    --prompt_embeddings prompt_embeddings.pt \
    --batch_size 4 \
    --num_workers 4

echo "[$(date +%T)] Done."
