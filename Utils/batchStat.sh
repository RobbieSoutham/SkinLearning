#!/bin/bash --login
#SBATCH -n 70
#SBATCH --job-name=Stats
#SBATCH -o stats.log
#SBATCH -e stats.err
#SBATCH -t 72:00:00
#SBATCH -p compute
srun python3 Stats.py

