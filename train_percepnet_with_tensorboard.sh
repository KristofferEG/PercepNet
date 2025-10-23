#!/bin/bash

# SLURM job script for PercepNet training with TensorBoard integration
# This script starts TensorBoard in the background for monitoring training progress
# TensorBoard will be accessible via SSH port forwarding on port 6006

#SBATCH --job-name=percepnet_training_tb
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
# TensorBoard configuration
#####################################################################################

# TensorBoard configuration
TENSORBOARD_PORT=6006
TENSORBOARD_LOGDIR="/ceph/home/student.aau.dk/ar01mf/PercepNet/training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99"
TENSORBOARD_PID_FILE="/tmp/tensorboard_${SLURM_JOB_ID}.pid"

# Function to start TensorBoard
start_tensorboard() {
    echo "Starting TensorBoard on port ${TENSORBOARD_PORT}"
    echo "TensorBoard log directory: ${TENSORBOARD_LOGDIR}"
    
    # Start TensorBoard in the background within the container
    singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif \
        tensorboard --logdir="${TENSORBOARD_LOGDIR}" \
        --port=${TENSORBOARD_PORT} \
        --host=0.0.0.0 \
        --reload_interval=30 &
    
    # Save the PID for cleanup
    TENSORBOARD_PID=$!
    echo $TENSORBOARD_PID > $TENSORBOARD_PID_FILE
    
    echo "TensorBoard started with PID: $TENSORBOARD_PID"
    echo "Access TensorBoard at: http://$(hostname):${TENSORBOARD_PORT}"
    echo "Note: You may need to set up port forwarding to access from your local machine"
}

# Function to stop TensorBoard
stop_tensorboard() {
    if [ -f "$TENSORBOARD_PID_FILE" ]; then
        TENSORBOARD_PID=$(cat $TENSORBOARD_PID_FILE)
        if kill -0 $TENSORBOARD_PID 2>/dev/null; then
            echo "Stopping TensorBoard (PID: $TENSORBOARD_PID)"
            kill $TENSORBOARD_PID
            # Wait a bit for graceful shutdown
            sleep 2
            # Force kill if still running
            if kill -0 $TENSORBOARD_PID 2>/dev/null; then
                echo "Force killing TensorBoard"
                kill -9 $TENSORBOARD_PID
            fi
        fi
        rm -f $TENSORBOARD_PID_FILE
    fi
}

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
    
    # Stop TensorBoard before requeuing or exiting
    stop_tensorboard
    
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
# Also trap EXIT to ensure TensorBoard cleanup on normal exit
trap 'stop_tensorboard' EXIT
echo "Signal traps set for SIGTERM and EXIT"

#####################################################################################

# Change to the PercepNet directory
cd /ceph/home/student.aau.dk/ar01mf/PercepNet

# Set up some environment variables
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=15
export MKL_NUM_THREADS=15
export CUDA_VISIBLE_DEVICES=0

echo "Starting PercepNet training with TensorBoard at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo ""
echo "================================================================================"
echo "TENSORBOARD ACCESS INFORMATION:"
echo "TensorBoard is running on: http://$(hostname):${TENSORBOARD_PORT}"
echo ""
echo "To access TensorBoard from your local machine (VS Code terminal):"
echo "1. Open a new terminal in VS Code and create an SSH tunnel:"
echo "   ssh -L ${TENSORBOARD_PORT}:$(hostname):${TENSORBOARD_PORT} -l AR01MF@student.aau.dk ailab-fe01.srv.aau.dk"
echo ""
echo "2. Keep the SSH connection open and open your browser to:"
echo "   http://localhost:${TENSORBOARD_PORT}"
echo ""
echo "Alternative method if direct forwarding doesn't work:"
echo "1. Connect to login node: ssh -l AR01MF@student.aau.dk ailab-fe01.srv.aau.dk"
echo "2. From login node run: ssh -L ${TENSORBOARD_PORT}:$(hostname):${TENSORBOARD_PORT} $(hostname)"
echo "3. Then use another VS Code terminal:"
echo "   ssh -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} -l AR01MF@student.aau.dk ailab-fe01.srv.aau.dk"
echo ""
echo "Compute node: $(hostname)"
echo "================================================================================"
echo ""

# Check GPU availability (use singularity directly for a quick check)
echo "Checking GPU availability:"
singularity exec --nv /ceph/home/student.aau.dk/ar01mf/PercepNet_Container.sif nvidia-smi || true

# Start TensorBoard before beginning training
start_tensorboard

# Give TensorBoard a moment to start up
sleep 5

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

# Stop TensorBoard when training is finished
stop_tensorboard

exit ${exit_code}