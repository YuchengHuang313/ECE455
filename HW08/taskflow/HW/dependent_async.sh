#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dependent_async.output

cd $SLURM_SUBMIT_DIR
module load gcc
g++ -std=c++20 ./HW/dependent_async.cpp -o ./HW/dependent_async -I. -pthread
./HW/dependent_async