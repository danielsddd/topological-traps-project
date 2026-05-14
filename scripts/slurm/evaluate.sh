#!/bin/bash
#SBATCH --job-name=traps_eval
#SBATCH --output=logs/slurm/eval/%j_eval.out
#SBATCH --error=logs/slurm/eval/%j_eval.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# =============================================================================
# SLURM Job Script: Model Evaluation
# =============================================================================
# This script evaluates a trained model:
# - Overall metrics on test set
# - Per-robot-size evaluation
# - Generalization to unseen sizes
# - Speed benchmarking
# - Visualization generation
#
# Usage:
#   sbatch scripts/slurm/evaluate.sh
#
# To evaluate a specific checkpoint:
#   sbatch scripts/slurm/evaluate.sh --checkpoint path/to/checkpoint.pth
# =============================================================================

echo "=========================================="
echo "EVALUATION JOB STARTED"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Environment setup
source ~/.bashrc

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate traps 2>/dev/null || echo "Note: conda env 'traps_env' not found"
fi

# Alternatively, activate virtualenv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Module loads
module load python/3.10 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Create log directory
mkdir -p logs/slurm

echo ""
echo "Environment:"
echo "  Python: $(which python)"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# Find best checkpoint
CHECKPOINT="${1:-outputs/checkpoints/best_iou.pth}"

# Look for the most recent experiment if default doesn't exist
if [ ! -f "$CHECKPOINT" ]; then
    LATEST_EXP=$(ls -td outputs/viability_* 2>/dev/null | head -1)
    if [ -n "$LATEST_EXP" ]; then
        CHECKPOINT="$LATEST_EXP/checkpoints/best_iou.pth"
    fi
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please specify a valid checkpoint path"
    exit 1
fi

echo ""
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "=========================================="
echo "STARTING EVALUATION"
echo "=========================================="

# Run evaluation
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config configs/config.yaml \
    --device cuda \
    --visualize \
    --benchmark-speed

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "EVALUATION COMPLETED SUCCESSFULLY"
    echo "=========================================="
else
    echo "EVALUATION FAILED (exit code: ${EXIT_CODE})"
    echo "=========================================="
fi

echo ""
echo "Job finished at: $(date)"

exit $EXIT_CODE
