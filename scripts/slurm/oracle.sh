#!/bin/bash
#SBATCH --job-name=trap_oracle
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/oracle/%j_oracle.out
#SBATCH --error=logs/slurm/oracle/%j_oracle.err

# ============================================================
# Oracle label generation job.
# CPU-only. Parallelized across maps via multiprocessing.
#
# Submit: sbatch scripts/slurm/oracle.sbatch
# Resume: sbatch scripts/slurm/oracle.sbatch  (auto-skips done)
# Force:  sbatch scripts/slurm/oracle.sbatch --force
# Test:   sbatch scripts/slurm/oracle.sbatch --max-maps 10
# ============================================================

source scripts/slurm/_header.sh

# ---- Data validation -------------------------------------------------------
echo "Validating input data..."

N_MAPS=$(ls "$PROCESSED_DIR"/*.npy 2>/dev/null | wc -l)
if [ "$N_MAPS" -eq 0 ]; then
    echo "ERROR: No processed maps in $PROCESSED_DIR"
    echo "Run: python scripts/preprocess.py first"
    exit 1
fi
echo "Processed maps : $N_MAPS"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found at $MANIFEST_PATH"
    exit 1
fi
echo "Manifest       : $MANIFEST_PATH ($(wc -l < $MANIFEST_PATH) lines)"

# Count existing labels
for SIZE_DIR in "$LABELS_DIR"/robot_*/; do
    if [ -d "$SIZE_DIR" ]; then
        N=$(ls "$SIZE_DIR"/*.npy 2>/dev/null | wc -l)
        echo "Existing labels: $(basename $SIZE_DIR) → $N"
    fi
done

echo "=================================================="

# ---- Build command ---------------------------------------------------------
CMD="python scripts/local/04_generate_labels.py \
    --config configs/config.yaml \
    --num-workers ${SLURM_CPUS_PER_TASK}"
    
# Pass through any extra args (--force, --max-maps N, etc.)
CMD="$CMD $@"

echo "COMMAND: $CMD"
echo "=================================================="

# ---- Run -------------------------------------------------------------------
$CMD
EXIT_CODE=$?

# ---- Report ----------------------------------------------------------------
echo "=================================================="
echo "EXIT CODE: $EXIT_CODE"
echo "END: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS"
    echo ""
    echo "Label counts:"
    for SIZE_DIR in "$LABELS_DIR"/robot_*/; do
        if [ -d "$SIZE_DIR" ]; then
            N=$(ls "$SIZE_DIR"/*.npy 2>/dev/null | wc -l)
            echo "  $(basename $SIZE_DIR): $N labels"
        fi
    done
    echo ""
    cat outputs/results/oracle_stats.json 2>/dev/null || true
else
    echo "FAILED — check logs/slurm/oracle/${SLURM_JOB_ID}_oracle.err"
    exit $EXIT_CODE
fi