#!/bin/bash
#SBATCH --job-name=small_matmul        
#SBATCH --partition=instruction                
#SBATCH --gres=gpu:1                   
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4             
#SBATCH --mem=4G                       
#SBATCH --time=00:10:00                
#SBATCH --output=slurm-%j.out          


module purge
module load gcc           
module load cuda          

cd "$SLURM_SUBMIT_DIR"

make clean
make

./small_matmul_test > run.log
