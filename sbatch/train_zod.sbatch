#!/bin/bash
#SBATCH --job-name=HC-Net_train
#SBATCH --chdir=/home/finckh/VITA/HC-Net
#SBATCH --account=civil-459
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20         # << ask for 20 CPUs
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=50G                 # << increased memory from 50G to 80GB
#SBATCH --time=2:00:00            # << set a realistic training time
#SBATCH --output=/home/finckh/VITA/HC-Net/logs/log_%j.log
#SBATCH --error=/home/finckh/VITA/HC-Net/logs/err_%j.log

# Make sure logs directory exists
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate dg

# Run training
python -u train.py --config /home/finckh/VITA/HC-Net/models/config/ZOD/train.json --batch_size 16  --name zod

conda deactivate
