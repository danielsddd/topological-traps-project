#!/bin/bash
#SBATCH --job-name=traps_train
#SBATCH --output=logs/slurm/train/%j_train.out
#SBATCH --error=logs/slurm/train/%j_train.err
#SBATCH --time=08:00:00
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# =============================================================================
# SLURM Job Script: Model Training
# =============================================================================
# This script trains the directional viability prediction model on the TAU cluster.
#
# Requirements:
# - Preprocessing must be completed first (run preprocess.sh)
# - GPU with at least 12GB memory recommended
#
# Usage:
#   sbatch scripts/slurm/train.sh
#
# To resume from checkpoint:
#   sbatch scripts/slurm/train.sh --resume outputs/checkpoints/last.pth
#
# Monitor training:
#   tail -f logs/slurm/train_<job_id>.out
#   tensorboard --logdir outputs/
# =============================================================================

echo "=========================================="
echo "TRAINING JOB STARTED"
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
echo "  Python version: $(python --version)"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi not available"

# Verify PyTorch CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available!"
    exit 1
fi

echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="

# Parse additional arguments passed to sbatch
RESUME_ARG=""
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    RESUME_ARG="--resume $2"
    echo "Resuming from checkpoint: $2"
fi

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
python scripts/train.py \
    --config configs/config.yaml \
    --device cuda \
    ${RESUME_ARG}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo ""
    echo "Results saved to: outputs/"
    echo ""
    echo "Next steps:"
    echo "  1. Review training curves in TensorBoard"
    echo "  2. Run evaluation: sbatch scripts/slurm/evaluate.sh"
else
    echo "TRAINING FAILED (exit code: ${EXIT_CODE})"
    echo "=========================================="
    echo ""
    echo "To resume training:"
    echo "  sbatch scripts/slurm/train.sh --resume outputs/checkpoints/last.pth"
fi

echo ""
echo "Job finished at: $(date)"

exit $EXIT_CODE
