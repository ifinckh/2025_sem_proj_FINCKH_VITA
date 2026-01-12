#!/bin/bash
#SBATCH --job-name=HC-Net_train
#SBATCH --chdir=/home/finckh/VITA/HC-Net
#SBATCH --account=civil-459
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/home/finckh/VITA/HC-Net/logs/log_%j.log
#SBATCH --error=/home/finckh/VITA/HC-Net/logs/err_%j.log

# Make sure logs directory exists
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate dg

# Run training
python -u myevaluate_zod.py --iters_lev0 6 --batch_size 16 --restore_ckpt checkpoints/2026_01_11_17_52_20_lastest_checkpoint_zod.pth --config models/config/ZOD/train.json

conda deactivate
