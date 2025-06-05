#!/bin/bash
#SBATCH --job-name=te_pipeline_100
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=pipeline_100_%j.out
#SBATCH --error=pipeline_100_%j.err

# Load required modules
module load python/3.9
module load openmpi/4.1.1

# Activate conda environment (adjust path as needed)
source activate idtxl_env

# Print environment information
echo "Job started at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Python version: $(python --version)"
echo "MPI version: $(mpirun --version)"

# Analysis parameters - using settings that worked well locally
N_NODES=5
N_SAMPLES=1000
N_REPLICATIONS=10
MAX_LAG=5
N_TRIALS=5
MODEL_TYPE="VAR"  # or "CLM"
SEED=42
N_JOBS=64  # Total jobs across all nodes (16 cores * 4 nodes)
SAVE_DIR="results_100_${MODEL_TYPE}"

# Create save directory
mkdir -p ${SAVE_DIR}

# Run the analysis using MPI
srun -N 4 -n 4 --mpi=pmi2 python -m mpi4py.futures pipeline_100.py \
    --n_nodes ${N_NODES} \
    --n_samples ${N_SAMPLES} \
    --n_replications ${N_REPLICATIONS} \
    --model_type ${MODEL_TYPE} \
    --max_lag ${MAX_LAG} \
    --n_trials ${N_TRIALS} \
    --seed ${SEED} \
    --n_jobs ${N_JOBS} \
    --save_dir ${SAVE_DIR}

# Print completion message
echo "Job finished at $(date)" 