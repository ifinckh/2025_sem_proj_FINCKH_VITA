#!/bin/bash
#SBATCH --job-name=dglsnet_train
#SBATCH --chdir=/home/finckh/VITA/D-GLSNet
#SBATCH --account=civil-459
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20         # << ask for 20 CPUs
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=100G                 # << increased memory from 50G to 80GB
#SBATCH --time=20:00:00            # << set a realistic training time
#SBATCH --output=logs/log_%j.log
#SBATCH --error=logs/err_%j.log

# Make sure logs directory exists
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate dg

# Run training
python dataloader_tester.py

conda deactivate
