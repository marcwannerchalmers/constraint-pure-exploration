#!/bin/bash -l
#SBATCH -A SLURM PROJECT NUMBER GOES HERE
#SBATCH -n 1
#SBATCH --output=exp.out
#SBATCH -t 03:00:00

conda activate ENVIRONMENT NAME HERE
srun --unbuffered python3 $1 $2 $3 $4
