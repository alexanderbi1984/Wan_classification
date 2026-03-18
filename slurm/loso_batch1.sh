#!/bin/bash
#SBATCH --job-name=LOSO_B1
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/LOSO_batch1_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=config_pain/config_lora_t100_aug3.yaml
BATCH_SIZE=20
OUT_DIR=results/loso
mkdir -p ${OUT_DIR}

# Batch 1: subjects 0–28 (29 folds)
SUBJECTS=(
    071309_w_21 071313_m_41 071614_m_20 071709_w_23 071814_w_23
    071911_w_24 072414_m_23 072514_m_27 072609_w_23 072714_m_23
    073109_w_28 073114_m_25 080209_w_26 080309_m_29 080314_w_25
    080609_w_27 080614_m_24 080709_m_24 080714_m_23 081014_w_27
    081609_w_40 081617_m_27 081714_m_36 082014_w_24 082109_m_53
    082208_w_45 082315_w_60 082414_m_64 082714_m_22
)

echo "[$(date)] LOSO Batch 1: ${#SUBJECTS[@]} subjects"
echo "Config: ${CONFIG}"
echo ""

COMPLETED=0
FAILED=0

for SUBJ in "${SUBJECTS[@]}"; do
    # Skip if already done
    if [ -f "${OUT_DIR}/${SUBJ}.json" ]; then
        echo "[$(date +%H:%M:%S)] ${SUBJ} — already completed, skipping"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo "========================================"
    echo "[$(date +%H:%M:%S)] Starting fold: ${SUBJ}"
    echo "========================================"

    torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 100)) \
        scripts/train_loso_fold.py \
        --config ${CONFIG} \
        --test_subject ${SUBJ} \
        --output_dir ${OUT_DIR} \
        --n_gpus 4 \
        --batch_size ${BATCH_SIZE}

    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "[$(date +%H:%M:%S)] ${SUBJ} — SUCCESS (${COMPLETED}/${#SUBJECTS[@]})"
    else
        FAILED=$((FAILED + 1))
        echo "[$(date +%H:%M:%S)] ${SUBJ} — FAILED (errors: ${FAILED})"
    fi
    echo ""
done

echo "========================================"
echo "[$(date)] Batch 1 complete: ${COMPLETED} succeeded, ${FAILED} failed"
echo "========================================"
