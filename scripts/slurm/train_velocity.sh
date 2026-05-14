#!/bin/bash
#SBATCH --job-name=train_velocity
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/train/velocity_%j.out
#SBATCH --error=logs/slurm/train/velocity_%j.err

# ============================================================
# Experiment 1: Velocity-Dependent Viability (Momentum Trap)
#
# oracle_type = velocity
# Input:  4 channels  (occ, L_norm, W_norm, v_norm)
# Output: 4 channels  (N, S, E, W binary viability)
# Loss:   Dice + BCE
#
# Velocities sampled during training: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 m/s
# Braking model: d_brake = v²/(2·a_max),  a_max = 2.0 m/s²
# Scale: 10 px/m
# ============================================================

set -euo pipefail

# Source shared header (conda activate, paths, etc.)
source scripts/slurm/_header.sh

echo "================================================================"
echo "Job:         ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Oracle type: velocity"
echo "================================================================"

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ---- Data validation ----
if [ ! -d "data/processed" ]; then
    echo "ERROR: data/processed/ not found"
    exit 1
fi
MAP_COUNT=$(ls data/processed/*.npy 2>/dev/null | wc -l)
echo "Maps found: ${MAP_COUNT}"
if [ "${MAP_COUNT}" -lt 100 ]; then
    echo "ERROR: fewer than 100 maps — aborting"
    exit 1
fi

if [ ! -f "data/manifest.csv" ]; then
    echo "ERROR: data/manifest.csv not found"
    exit 1
fi
echo "Manifest OK"

# ---- Patch train.py if not already done ----

# ---- Resume support ----
RESUME_FLAG=""
LATEST_CKPT=""
# Look for the most recent velocity experiment directory
for d in outputs/viability_velocity_*/checkpoints/last.pth; do
    if [ -f "$d" ]; then
        LATEST_CKPT="$d"
    fi
done

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from: ${LATEST_CKPT}"
    RESUME_FLAG="--resume ${LATEST_CKPT}"
fi

# ---- Train ----
python scripts/train.py \
    --config configs/config.yaml \
    --oracle_type velocity \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0001 \
    ${RESUME_FLAG} \
    2>&1 | tee logs/slurm/train/velocity_${SLURM_JOB_ID}_train.log

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Training finished with exit code: ${EXIT_CODE}"
echo "Date: $(date)"
echo "================================================================"

exit ${EXIT_CODE}
