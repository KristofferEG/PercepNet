#!/bin/bash

#SBATCH --job-name=percepnet_deploy
#SBATCH --output=deploy_%j.out
#SBATCH --error=deploy_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=00:30:00
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-01

# Change to the PercepNet directory
cd /ceph/home/student.aau.dk/ar01mf/PercepNet

# Set up some environment variables
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=15
export MKL_NUM_THREADS=15
export CUDA_VISIBLE_DEVICES=0

echo "Starting PercepNet deployment at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"

# Check GPU availability
echo "Checking GPU availability:"
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif nvidia-smi

# Step 1: Convert .pt checkpoint to .pkl format for compatibility
echo "Converting .pt checkpoint to .pkl format..."
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif \
    python3 -c "
import torch
import sys
sys.path.append('/ceph/home/student.aau.dk/ar01mf/PercepNet')

# Load the .pt checkpoint
checkpoint_path = 'training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99/checkpoint-100000steps.pt'
pkl_path = 'training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99/checkpoint-100000steps.pkl'

print(f'Loading checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract model_state_dict and save as .pkl
if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
    print(f'Extracted model_state_dict with {len(model_state_dict)} keys')
    
    # Save as .pkl format (direct state dict)
    torch.save(model_state_dict, pkl_path)
    print(f'Saved model weights to: {pkl_path}')
    print('Conversion completed successfully!')
else:
    print('Error: model_state_dict not found in checkpoint')
    sys.exit(1)
"

# Step 2: Dump weights from PyTorch to C++ header using the .pkl file
echo "Dumping PyTorch weights to C++ header..."
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif \
    python3 dump_percepnet.py training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99/checkpoint-100000steps.pkl

echo "Building C++ inference binary..."
cd bin
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif cmake ..
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif make -j1

cd ..

echo "Testing inference..."
if [ -f "sampledata/speech/speech.pcm" ]; then
    echo "✓ Test input file found: sampledata/speech/speech.pcm"
    echo "Input file size: $(ls -lh sampledata/speech/speech.pcm | awk '{print $5}')"
    
    echo "Running correct inference test with percepNet_run..."
    ./bin/src/percepNet_run sampledata/speech/speech.pcm deployment_output.pcm
    
    if [ $? -eq 0 ] && [ -f "deployment_output.pcm" ]; then
        echo "✓ Inference test successful!"
        echo "Output file size: $(ls -lh deployment_output.pcm | awk '{print $5}')"
        echo "Input vs Output comparison:"
        echo "  Input:  $(ls -lh sampledata/speech/speech.pcm | awk '{print $5, $9}')"
        echo "  Output: $(ls -lh deployment_output.pcm | awk '{print $5, $9}')"
    else
        echo "✗ Inference test failed"
        exit 1
    fi
else
    echo "⚠ Sample speech file not found - skipping inference test"
fi

echo "Deployment completed at $(date)"
echo "Exit code: $?"