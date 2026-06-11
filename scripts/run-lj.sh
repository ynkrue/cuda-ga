#!/bin/bash
#SBATCH --job-name=lj
#SBATCH --output=logs/lj_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1

srun bin/sa --config configs/lj.ini
