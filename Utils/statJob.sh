#!/bin/bash --login
###
#SBATCH --job_name=Stats
#SBATCH --output=stats.out.%J
#SBATCH --err=stats.err.%J
#SBATCH --time=3-00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu
