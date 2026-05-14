#!/bin/bash
#SBATCH --job-name=traps_cost_map
#SBATCH --output=logs/slurm/train/%j_cost_map.out
#SBATCH --error=logs/slurm/train/%j_cost_map.err
#SBATCH --time=08:00:00
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# =============================================================================
# SLURM Job Script: Cost Map (Escape Distance Regression) Training
# =============================================================================
# Trains a 3-channel U-Net to predict how far each pixel is from safety
# in each cardinal direction (regression extension of binary viability).
# Input:  (occupancy, robot_L, robot_W) — 3 channels (same as basic)
# Output: 4-channel escape distance map (N/S/E/W) — continuous, not binary
# Loss:   SmoothL1Loss (regression)
#
# Usage:
#   sbatch scripts/slurm/train_cost_map.sh
#
# Auto-resumes from the latest cost_map checkpoint if one exists.
# Re-submit the same command after preemption — it will pick up where it left off.
# =============================================================================

echo "=========================================="
echo "COST MAP TRAINING JOB STARTED"
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
RESUME_ARG=""
LATEST_CKPT=$(ls -td outputs/viability_cost_map_*/checkpoints/last.pth 2>/dev/null | head -1)

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
echo "STARTING TRAINING (oracle_type=cost_map)"
echo "=========================================="

python scripts/train.py \
    --config configs/config.yaml \
    --oracle_type cost_map \
    --epochs 30 \
    --device cuda \
    ${RESUME_ARG}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    BEST=$(ls -td outputs/viability_cost_map_*/checkpoints/best_iou.pth 2>/dev/null | head -1)
    echo "Best checkpoint: ${BEST}"
    echo ""
    echo "Next: run zero-shot and demo evaluations with this checkpoint"
else
    echo "TRAINING FAILED OR PREEMPTED (exit code: ${EXIT_CODE})"
    echo "=========================================="
    echo ""
    echo "Re-submit to resume:"
    echo "  sbatch scripts/slurm/train_cost_map.sh"
    echo "(script will auto-detect and resume from last.pth)"
fi

echo ""
echo "Job finished at: $(date)"
exit $EXIT_CODE