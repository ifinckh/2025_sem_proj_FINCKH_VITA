#!/bin/bash
#SBATCH --job-name=dglsnet_test_zod
#SBATCH --chdir /home/finckh/VITA/D-GLSNet
#SBATCH --ntasks=1
#SBATCH --account=civil-459
#SBATCH --cpus-per-task=20       # << ask for 20 CPUs
#SBATCH --gres=gpu:1
#SBATCH --mem=50G                         
#SBATCH --time=00:40:00
#SBATCH --output=logs/log_%j.log
#SBATCH --error=logs/err_%j.log

source ~/.bashrc
conda activate dg
CUDA_VISIBLE_DEVICES=0 python train.py pl_DGLSNet ./config/default_zod.yaml zod_test --test
conda deactivate