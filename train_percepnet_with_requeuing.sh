#!/bin/bash

#SBATCH --job-name=percepnet_training
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-01
# Allow the job to be requeued by Slurm when possible
# (depends on cluster configuration and admin settings)
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30

#####################################################################################
# Requeuing configuration
#####################################################################################

# tweak this to fit your needs
max_restarts=4

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Dynamically set output and error filenames using job ID and iteration
outfile="training_${SLURM_JOB_ID}_${iteration}.out"
errfile="training_${SLURM_JOB_ID}_${iteration}.err"

# Print the filenames for debugging
echo "Output file: ${outfile}"
echo "Error file: ${errfile}"

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    echo "Current restart count: $restarts"
    echo "Max restarts: $max_restarts"
    
    if [[ $restarts -lt $max_restarts ]]; then
        echo "Requeuing job ${SLURM_JOB_ID}..."
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM
echo "Signal trap set for SIGTERM"

#####################################################################################

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

# Check GPU availability (use singularity directly for a quick check)
echo "Checking GPU availability:"
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif nvidia-smi || true

# Run the training under srun so Slurm signals (SIGTERM/USR1) are
# delivered to the job step and our trap can requeue the job when needed.
# Use the dynamic outfile/errfile names computed above.
cd utils
echo "Running PercepNet training using run.sh (Stage 4)"

# Check checkpoint status before starting
echo "Checking for existing checkpoints..."
CHECKPOINT_DIR="/ceph/home/student.aau.dk/ar01mf/PercepNet/training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99"
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/checkpoint-*steps.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Latest checkpoint found: $LATEST_CHECKPOINT"
    else
        echo "No .pt checkpoints found, checking for .pkl files..."
        LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/checkpoint-*steps.pkl 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Latest checkpoint found: $LATEST_CHECKPOINT"
        else
            echo "No existing checkpoints found - starting from scratch"
        fi
    fi
else
    echo "Checkpoint directory does not exist - will be created during training"
fi

# Launch the container under srun and capture stdout/stderr into iteration-specific files
srun --output="${outfile}" --error="${errfile}" singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif ./run.sh

exit_code=$?
echo "Training completed at $(date)"
echo "Exit code: ${exit_code}"
exit ${exit_code}