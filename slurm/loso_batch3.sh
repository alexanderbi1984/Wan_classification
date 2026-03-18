#!/bin/bash
#SBATCH --job-name=LOSO_B3
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/LOSO_batch3_%j.out

source /opt/ohpc/pub/apps/miniforge3/etc/profile.d/conda.sh
conda activate wan
cd /data/home/nbi1/Wan_classification

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=config_pain/config_lora_t100_aug3.yaml
BATCH_SIZE=20
OUT_DIR=results/loso
mkdir -p ${OUT_DIR}

# Batch 3: subjects 58–86 (29 folds)
SUBJECTS=(
    101514_w_36 101609_m_36 101809_m_59 101814_m_58 101908_m_61
    101916_m_40 102008_w_22 102214_w_36 102309_m_61 102316_w_50
    102414_w_58 102514_w_40 110614_m_42 110810_m_62 110909_m_29
    111313_m_64 111409_w_63 111609_m_65 111914_w_63 112009_w_43
    112016_m_25 112209_m_51 112310_m_20 112610_w_60 112809_w_23
    112909_w_20 112914_w_51 120514_w_56 120614_w_61
)

echo "[$(date)] LOSO Batch 3: ${#SUBJECTS[@]} subjects"
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

    torchrun --nproc_per_node=4 --master_port=$((29700 + RANDOM % 100)) \
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
echo "[$(date)] Batch 3 complete: ${COMPLETED} succeeded, ${FAILED} failed"
echo "========================================"
