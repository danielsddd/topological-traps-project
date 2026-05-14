#!/bin/bash
#SBATCH --job-name=traps_continuous_angle
#SBATCH --output=logs/slurm/train/%j_continuous_angle.out
#SBATCH --error=logs/slurm/train/%j_continuous_angle.err
#SBATCH --time=08:00:00
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# =============================================================================
# SLURM Job Script: Continuous-Angle Viability Training
# =============================================================================
# Trains a 5-channel U-Net that predicts viability for arbitrary heading angles.
# Input:  (occupancy, robot_L, robot_W, sin(θ), cos(θ)) — 5 channels
# Output: single viability mask for the given heading angle
#
# Usage:
#   sbatch scripts/slurm/train_continuous_angle.sh
#
# Auto-resumes from the latest continuous_angle checkpoint if one exists.
# Re-submit the same command after preemption — it will pick up where it left off.
# =============================================================================

echo "=========================================="
echo "CONTINUOUS ANGLE TRAINING JOB STARTED"
echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Time:     $(date)"
echo "Workdir:  $(pwd)"
echo "=========================================="

# ---- Environment setup -------------------------------------------------------
source ~/.bashrc
conda activate traps 2>/dev/null || echo "WARNING: conda env 'traps' not found"

mkdir -p logs/slurm/train

# ---- Verify GPU --------------------------------------------------------------
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "nvidia-smi not available"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available — aborting"
    exit 1
fi

# ---- Auto-detect resume checkpoint -------------------------------------------
# Find the latest continuous_angle experiment that has a last.pth checkpoint.
# If none found, starts a fresh run.
RESUME_ARG=""
LATEST_CKPT=$(ls -td outputs/viability_continuous_angle_*/checkpoints/last.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo ""
    echo "Found existing checkpoint: ${LATEST_CKPT}"
    echo "Resuming training..."
    RESUME_ARG="--resume ${LATEST_CKPT}"
else
    echo ""
    echo "No existing checkpoint found — starting fresh run"
fi

# ---- Environment variables ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ---- Run training ------------------------------------------------------------
echo ""
echo "=========================================="
echo "STARTING TRAINING (oracle_type=continuous_angle)"
echo "=========================================="

python scripts/train.py \
    --config configs/config.yaml \
    --oracle_type continuous_angle \
    --epochs 30 \
    --num-angles-per-map 8 \
    --device cuda \
    ${RESUME_ARG}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    # Print best checkpoint location
    BEST=$(ls -td outputs/viability_continuous_angle_*/checkpoints/best_iou.pth 2>/dev/null | head -1)
    echo "Best checkpoint: ${BEST}"
    echo ""
    echo "Next: run zero-shot and demo evaluations with this checkpoint"
else
    echo "TRAINING FAILED OR PREEMPTED (exit code: ${EXIT_CODE})"
    echo "=========================================="
    echo ""
    echo "Re-submit to resume:"
    echo "  sbatch scripts/slurm/train_continuous_angle.sh"
    echo "(script will auto-detect and resume from last.pth)"
fi

echo ""
echo "Job finished at: $(date)"
exit $EXIT_CODE