#!/bin/bash
source configs/.env
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

echo "=================================================="
echo "JOB:        $SLURM_JOB_NAME"
echo "JOB ID:     $SLURM_JOB_ID"
echo "NODE:       $(hostname)"
echo "START:      $(date)"
echo "PROJECT:    $PROJECT_DIR"
echo "CONDA ENV:  $CONDA_ENV"
echo "GIT HASH:   $(git rev-parse --short HEAD 2>/dev/null || echo N/A)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-unknown}"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo "=================================================="