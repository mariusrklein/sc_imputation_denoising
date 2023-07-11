#!/bin/bash

#SBATCH --job-name=imp_eval
#SBATCH --output=/home/mklein/cluster_jobs/imp_eval_%j_output.log
#SBATCH --error=/home/mklein/cluster_jobs/imp_eval_%j_error.log
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=48:00:00

# arguments: dataset max_fdr n_cells n_ions i

echo "Starting job $SLURM_JOB_ID with arguments dataset=$1, sim=$2, repl=$3"

# Activate your conda environment if necessary
. /home/mklein/miniconda3/etc/profile.d/conda.sh 
conda activate /home/mklein/miniconda3/envs/env_all

# Navigate to the directory containing the notebook
cd /home/mklein/Dropouts/evaluation

# Run the notebook using papermill and save output to new notebook file
papermill Imputation_evaluation_mse.ipynb ~/cluster_jobs/$1.$2.$3.$SLURM_JOB_ID.out.ipynb -k env_all --log-output --report-mode -p dataset $1 -p simulation_method $2 -p repl $3
# papermill add_iterative.ipynb ~/cluster_jobs/$1.$2.$3.$SLURM_JOB_ID.out.ipynb -k env_all --log-output --report-mode -p dataset $1 -p simulation_method $2 -p repl $3

echo "Ending job $SLURM_JOB_ID with arguments dataset=$1, sim=$2, repl=$3"