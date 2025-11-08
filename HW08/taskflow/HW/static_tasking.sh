#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=static_tasking.output

cd $SLURM_SUBMIT_DIR
module load gcc
g++ -std=c++20 ./HW/static_tasking.cpp -o ./HW/static_tasking -I. -pthread
./HW/static_tasking
