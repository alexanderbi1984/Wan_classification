#!/bin/bash
#SBATCH --job-name=LOSO_B2
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/LOSO_batch2_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=config_pain/config_lora_t100_aug3.yaml
BATCH_SIZE=20
OUT_DIR=results/loso
mkdir -p ${OUT_DIR}

# Batch 2: subjects 29–57 (29 folds)
SUBJECTS=(
    082809_m_26 082814_w_46 082909_m_47 083009_w_42 083013_w_47
    083109_m_60 083114_w_55 091809_w_43 091814_m_37 091914_m_46
    092009_m_54 092014_m_56 092509_w_51 092514_m_50 092714_m_64
    092808_m_51 092813_w_24 100117_w_36 100214_m_50 100417_m_44
    100509_w_43 100514_w_51 100909_w_65 100914_m_39 101015_w_43
    101114_w_37 101209_w_61 101216_m_40 101309_m_48
)

echo "[$(date)] LOSO Batch 2: ${#SUBJECTS[@]} subjects"
echo "Config: ${CONFIG}"
echo ""

COMPLETED=0
FAILED=0

for SUBJ in "${SUBJECTS[@]}"; do
    if [ -f "${OUT_DIR}/${SUBJ}.json" ]; then
        echo "[$(date +%H:%M:%S)] ${SUBJ} — already completed, skipping"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo "========================================"
    echo "[$(date +%H:%M:%S)] Starting fold: ${SUBJ}"
    echo "========================================"

    torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 100)) \
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
echo "[$(date)] Batch 2 complete: ${COMPLETED} succeeded, ${FAILED} failed"
echo "========================================"
