#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/train_novisdom.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U10T12S1S1_O30_r100each \
--name=S1S1_U10_condi_sameD_addtoall \
--model=BiCycleGAN \
--batch_size=64 \
--input_nc=3 \
--output_nc=3 \
--dataset_mode=S1S1 \
--save_epoch_freq=1 \
--batch=instance \
--gan_mode=lsgan \
--num_threads=8 \
--lr=4e-4 \
--niter=50 \
--niter_decay=150 
exit