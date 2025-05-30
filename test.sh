#!/bin/bash
#SBATCH -p general-gpu
#SBATCH --constraint="rtx"
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "Running SLURM job $SLURM_JOB_ID on node $SLURM_NODELIST"