#!/bin/bash

# run every job 5 times to get better robustness insights.
num_jobs=5

declare -a datasets=("Lx_HepaRG" "Lx_Pancreatic_Cancer" "Lx_Glioblastoma" "Mx_Seahorse")
declare -a sims=("mnar")

# Loop over the jobs and submit them to SLURM
for sim in "${sims[@]}"; do
    for dataset in "${datasets[@]}"; do
        for i in $(seq 1 $num_jobs); do
            sleep 1
            echo "Submitting job $i for $dataset with $sim"
            # trigger a job for the given dataset, simulation method and replicate
            sbatch imp_eval.sh $dataset $sim $i
        done
    done
done
