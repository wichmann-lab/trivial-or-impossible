#!/bin/bash

#SBATCH --ntasks=1               # Number of tasks (see below)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=10-00:00            # Runtime in D-HH:MM
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=out/ResNet%j.out  # File to which STDOUT will be written
#SBATCH --error=out/ResNet%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --constraint=ImageNet2012   # Constrain to nodes where ImageNet is quickly available
#SBATCH --partition=gpu-2080ti-long

# Print info about current job
scontrol show job $SLURM_JOB_ID

python3 main.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}
