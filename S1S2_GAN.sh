#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/train_novisdom.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U10T12sortNoCloud \
--name=S1S2_U10NoCloud \
--model=pix2pix \
--batch_size=32 \
--input_nc=9 \
--netD=basic \
--netG=unet_256 \
--dataset_mode=S2S1 \
--save_epoch_freq=1 \
--batch=instance \
--gan_mode=wgangp \
--num_threads=8 \
--lr=1e-4 \
--niter=100 \
--niter_decay=200 
exit