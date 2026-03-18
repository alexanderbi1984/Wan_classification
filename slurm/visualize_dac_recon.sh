#!/bin/bash
#SBATCH --job-name=DaC_Vis
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/DaC_Vis_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CKPT_DIR=Wan2.1-T2V-14B
CONFIG=config_pain/config_lora_t100_aug3.yaml
OUT_DIR=results/dac/reconstructions

# Select specific interesting examples:
#   - PA4 correct: 080309_m_29-PA4-057_aligned
#   - PA1 correct: 080309_m_29-PA1-054_aligned
#   - PA4 wrong (pred=BL1): 080309_m_29-PA4-002_aligned
#   - PA3 correct: 080309_m_29-PA3-048_aligned
#   - BL1 correct: 080309_m_29-BL1-082_aligned
#   - PA2 wrong (pred=BL1): 080309_m_29-PA2-010_aligned
VIDEOS="080309_m_29-PA4-057_aligned,080309_m_29-PA1-054_aligned,080309_m_29-PA4-002_aligned,080309_m_29-PA3-048_aligned,080309_m_29-BL1-082_aligned,080309_m_29-PA2-010_aligned"

python scripts/visualize_dac_reconstruction.py \
    --checkpoint_dir ${CKPT_DIR} \
    --config ${CONFIG} \
    --output_dir ${OUT_DIR} \
    --video_ids "${VIDEOS}" \
    --results_json results/dac/dac_results.json \
    --timesteps 200 500 800

echo "[$(date)] DaC visualization complete"
