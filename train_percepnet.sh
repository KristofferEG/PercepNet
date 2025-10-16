#!/bin/bash

#SBATCH --job-name=percepnet_training
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=l4

# Change to the PercepNet directory
cd /ceph/home/student.aau.dk/ar01mf/PercepNet

# Set up some environment variables
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=15
export MKL_NUM_THREADS=15
export CUDA_VISIBLE_DEVICES=0

echo "Starting PercepNet training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"

# Check GPU availability
echo "Checking GPU availability:"
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif nvidia-smi

# Run the simple training function using the container
# This uses the h5Dataset with the training.h5 file we created
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif \
    python3 -c "
import sys
import torch
sys.path.append('/ceph/home/student.aau.dk/ar01mf/PercepNet')

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available! Training will be slow on CPU.')

"
# Change to utils directory and run the proper run.sh script
cd utils
echo "Running PercepNet training using run.sh (Stage 4)"
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif ./run.sh

echo "Training completed at $(date)"
echo "Exit code: $?"