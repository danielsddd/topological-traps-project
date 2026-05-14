#!/bin/bash
#SBATCH --job-name=traps_preprocess
#SBATCH --output=logs/slurm/preprocess_%j.out
#SBATCH --error=logs/slurm/preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# =============================================================================
# SLURM Job Script: Data Preprocessing
# =============================================================================
# This script runs the preprocessing pipeline on the TAU cluster:
# 1. Convert HouseExpo JSON maps to occupancy grids
# 2. Generate viability labels for all robot sizes
# 3. Create train/val/test manifest
#
# Usage:
#   sbatch scripts/slurm/preprocess.sh
#
# After completion, run training with:
#   sbatch scripts/slurm/train.sh
# =============================================================================

echo "=========================================="
echo "PREPROCESSING JOB STARTED"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Environment setup
source ~/.bashrc

# Activate conda environment (adjust as needed)
if command -v conda &> /dev/null; then
    conda activate traps_env 2>/dev/null || echo "Note: conda env 'traps_env' not found"
fi

# Alternatively, activate virtualenv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Module loads (adjust based on cluster configuration)
module load python/3.10 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Create log directory
mkdir -p logs/slurm

# Set number of workers based on available CPUs
export NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Workers: ${NUM_WORKERS}"
echo ""

# Verify dependencies
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || exit 1
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || exit 1

echo ""
echo "=========================================="
echo "RUNNING PREPROCESSING"
echo "=========================================="

# Run preprocessing
python scripts/preprocess.py \
    --config configs/config.yaml \
    --num-workers ${NUM_WORKERS} \
    --verify

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "PREPROCESSING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Review the verification output above"
    echo "  2. Submit training job: sbatch scripts/slurm/train.sh"
else
    echo "PREPROCESSING FAILED (exit code: ${EXIT_CODE})"
    echo "=========================================="
    echo ""
    echo "Check the log files for errors:"
    echo "  logs/slurm/preprocess_${SLURM_JOB_ID}.out"
    echo "  logs/slurm/preprocess_${SLURM_JOB_ID}.err"
fi

echo ""
echo "Job finished at: $(date)"

exit $EXIT_CODE
