#!/bin/bash -l
#SBATCH --job-name=T5_prompt_precompute
#SBATCH --partition=gpucompute-h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

module load miniconda
eval "$(/opt/ohpc/pub/apps/miniforge3/bin/conda shell.bash hook)"
conda activate wan || { echo "conda activate wan failed"; exit 1; }

cd ~/Wan_classification

python scripts/precompute_prompt_embeddings.py \
    --checkpoint_dir Wan2.1-T2V-14B \
    --output prompt_embeddings.pt \
    --device auto

echo "[$(date +%T)] Done."
