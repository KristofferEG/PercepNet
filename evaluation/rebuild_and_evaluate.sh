#!/bin/bash
#SBATCH --job-name=rebuild_100k
#SBATCH --output=rebuild_eval_%j.out
#SBATCH --error=rebuild_eval_%j.err
#SBATCH --partition=batch
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Rebuild PercepNet with 100k checkpoint and evaluate with PESQ

set -e

PERCEPNET_DIR="/ceph/home/student.aau.dk/ar01mf"
CONTAINER="${PERCEPNET_DIR}/PercepNet_Container.sif"
CHECKPOINT="${PERCEPNET_DIR}/training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99/checkpoint-100000steps.pkl"

echo "=== Step 1: Convert 100k checkpoint to C ==="
cd ${PERCEPNET_DIR}
singularity exec ${CONTAINER} python dump_percepnet.py ${CHECKPOINT} nnet_data.c
echo "✓ Checkpoint converted to nnet_data.c"

echo ""
echo "=== Step 2: Rebuild C++ binary ==="
cd ${PERCEPNET_DIR}/bin
singularity exec ${CONTAINER} make clean
singularity exec ${CONTAINER} make -j8
echo "✓ Binary rebuilt with 100k checkpoint"

echo ""
echo "=== Step 3: Test the new model ==="
# Run inference on a test file
TEST_INPUT="${PERCEPNET_DIR}/sampledata/speech/speech.pcm"
TEST_OUTPUT="${PERCEPNET_DIR}/deployment_outputs/output_100k.pcm"

cd ${PERCEPNET_DIR}/bin/src
singularity exec ${CONTAINER} ./percepNet ${TEST_INPUT} ${TEST_OUTPUT}
echo "✓ Inference completed: ${TEST_OUTPUT}"

echo ""
echo "=== Step 4: Evaluate with PESQ + STOI + DNSMOS ==="
# Convert PCM to WAV for evaluation
cd ${PERCEPNET_DIR}
singularity exec ${CONTAINER} python -c "
import soundfile as sf
import numpy as np
enhanced = np.fromfile('${TEST_OUTPUT}', dtype=np.int16).astype(np.float32) / 32768.0
sf.write('deployment_outputs/output_100k.wav', enhanced, 48000)
print('✓ Converted to WAV at 48kHz')
"

# Run full evaluation (will work because container has PESQ)
cd ${PERCEPNET_DIR}/evaluation
singularity exec ${CONTAINER} python evaluate.py \
    --enhanced ../deployment_outputs/output_100k.wav \
    --noisy ../sampledata/speech/speech.pcm \
    --format pcm \
    --sr 48000 \
    --output-dir ../evaluation_results \
    --save-plots

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: ${PERCEPNET_DIR}/evaluation_results/results.txt"
cat ${PERCEPNET_DIR}/evaluation_results/results.txt

echo ""
echo "=== Checkpoint Info ==="
echo "Model: checkpoint-100000steps.pkl"
echo "Training steps: 100,000"
echo "Date: $(stat -c %y ${CHECKPOINT} 2>/dev/null || echo 'Unknown')"
