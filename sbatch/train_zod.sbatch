#!/bin/bash
#SBATCH --job-name=HC-Net_train
#SBATCH --chdir=/home/finckh/VITA/HC-Net
#SBATCH --account=civil-459
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=70:00:00
#SBATCH --output=/home/finckh/VITA/HC-Net/logs/log_%j.log
#SBATCH --error=/home/finckh/VITA/HC-Net/logs/err_%j.log

# Make sure logs directory exists
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate dg

# Run training
python -u train.py --config /home/finckh/VITA/HC-Net/models/config/ZOD/train.json  --name zod_train # --batch_size 16
# python -u train.py --config /home/finckh/VITA/HC-Net/models/config/ZOD/train.json --name zod_train --restore_ckpt checkpoints/2026_01_09_14_36_50_best_checkpoint_zod.pth

conda deactivate
