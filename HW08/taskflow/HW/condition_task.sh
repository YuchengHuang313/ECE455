#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR
module load gcc
g++ -std=c++20 ./HW/condition_task.cpp -o ./HW/condition_task -I. -pthread
./HW/condition_task
