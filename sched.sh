#!/bin/bash --login
###
#job name
#SBATCH --job-name=imb_bench
#job stdout file
#SBATCH --output=bench.out.%J
#job stderr file
#SBATCH --error=bench.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=3-00:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=1 
#memory per process in MB 
#SBATCH --mem-per-cpu=300 
#tasks to run per node (change for hybrid OpenMP/MPI) 
#SBATCH --ntasks-per-node=1
###

#now run normal batch commands 
module load compute

while true
do
    python3 purge.py
    sleep 1h
done