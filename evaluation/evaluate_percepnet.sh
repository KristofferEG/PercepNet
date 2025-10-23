#!/bin/bash

#SBATCH --job-name=percepnet_eval
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=l4

# PercepNet Evaluation Script
# Evaluates the enhanced audio output against clean reference

# Change to the PercepNet directory
cd /ceph/home/student.aau.dk/ar01mf/PercepNet

# Configuration
CONTAINER="/ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif"

# Input files (relative to PercepNet directory)
ENHANCED_FILE="deployment_outputs/deployment_output.pcm"
NOISY_FILE="sampledata/speech/speech.pcm"
OUTPUT_DIR="evaluation_results"

echo "========================================"
echo "PercepNet Non-Intrusive Evaluation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo ""
echo "Enhanced file: $ENHANCED_FILE"
echo "Noisy file:    $NOISY_FILE (for comparison plots)"
echo "Output dir:    $OUTPUT_DIR"
echo "Sample rate:   48 kHz (PercepNet native)"
echo "========================================"
echo ""

if [ ! -f "$ENHANCED_FILE" ]; then
    echo "ERROR: Enhanced file not found: $ENHANCED_FILE"
    echo "Please run deployment first to generate enhanced audio"
    exit 1
fi

echo "✓ All input files found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation using Singularity container
echo "Running evaluation with Singularity container..."
echo ""

singularity exec --nv "$CONTAINER" python3 evaluation/evaluate.py \
    --enhanced "$ENHANCED_FILE" \
    --noisy "$NOISY_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --format pcm \
    --sr 48000 \
    --save-plots

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✓ Evaluation completed successfully!"
    echo "========================================"
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    
    # Display results if available
    if [ -f "$OUTPUT_DIR/results.txt" ]; then
        echo "Results:"
        cat "$OUTPUT_DIR/results.txt"
    fi
    
    # List output files
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR/"
else
    echo "========================================"
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
    echo "========================================"
fi

echo ""
echo "Evaluation completed at $(date)"
exit $EXIT_CODE
