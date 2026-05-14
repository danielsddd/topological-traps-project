#!/bin/bash
#SBATCH --job-name=eval_experiments
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval/experiments_%j.out
#SBATCH --error=logs/slurm/eval/experiments_%j.err

# ============================================================
# Evaluate both new experiments:
#   1. Velocity-Dependent Viability (Momentum Trap)
#   2. Time-to-Escape Cost Maps (Continuous Regression)
#
# Expects checkpoints to already exist from training runs.
# ============================================================

set -euo pipefail
source scripts/slurm/_header.sh

echo "================================================================"
echo "Job:   ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "Date:  $(date)"
echo "Task:  Evaluate velocity + cost_map experiments"
echo "================================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- Find checkpoints ----
# Velocity model
VEL_CKPT=$(ls -t outputs/viability_velocity_*/checkpoints/best_iou.pth 2>/dev/null | head -1)
if [ -z "$VEL_CKPT" ]; then
    echo "WARNING: No velocity checkpoint found — skipping Experiment 1"
else
    echo "Velocity checkpoint: ${VEL_CKPT}"
fi

# Cost-map model
COST_CKPT=$(ls -t outputs/viability_cost_map_*/checkpoints/best_iou.pth 2>/dev/null | head -1)
if [ -z "$COST_CKPT" ]; then
    echo "WARNING: No cost_map checkpoint found — skipping Experiment 2"
else
    echo "Cost-map checkpoint: ${COST_CKPT}"
fi

# Basic model (for comparison)
BASIC_CKPT=$(ls -t outputs/viability_basic_*/checkpoints/best_iou.pth 2>/dev/null | \
             head -1 || \
             ls -t outputs/viability_2026*/checkpoints/best_iou.pth 2>/dev/null | head -1)

echo ""

# ---- Experiment 1: Velocity Evaluation ----
if [ -n "$VEL_CKPT" ]; then
    echo "================================================================"
    echo "EXPERIMENT 1: Velocity-Dependent Viability"
    echo "================================================================"

    python scripts/evaluate_velocity.py \
        --checkpoint "${VEL_CKPT}" \
        --config configs/config.yaml \
        --output-dir outputs/velocity_evaluation \
        --velocities 0.0 0.5 1.0 1.5 2.0 2.5 3.0 \
        --robot-size 30 20 \
        --num-maps 50 \
        2>&1 | tee logs/slurm/eval/velocity_${SLURM_JOB_ID}.log

    echo ""
    echo "Velocity evaluation complete."
    echo ""
fi

# ---- Experiment 2: Cost Map Evaluation ----
if [ -n "$COST_CKPT" ]; then
    echo "================================================================"
    echo "EXPERIMENT 2: Time-to-Escape Cost Maps"
    echo "================================================================"

    BASIC_FLAG=""
    if [ -n "$BASIC_CKPT" ]; then
        BASIC_FLAG="--basic-checkpoint ${BASIC_CKPT}"
    fi

    python scripts/evaluate_cost_map.py \
        --checkpoint "${COST_CKPT}" \
        --config configs/config.yaml \
        --output-dir outputs/cost_map_evaluation \
        --robot-size 30 20 \
        --num-maps 50 \
        ${BASIC_FLAG} \
        2>&1 | tee logs/slurm/eval/cost_map_${SLURM_JOB_ID}.log

    echo ""
    echo "Cost-map evaluation complete."
    echo ""
fi

echo "================================================================"
echo "All evaluations finished."
echo "Date: $(date)"
echo "================================================================"
