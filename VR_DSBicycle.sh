#!/bin/bash
#SBATCH -p general-gpu
#SBATCH --constraint="a100"
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/train_novisdom.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Downscaling/VIIRs_RAPID/Samples/RAPIDonly_PerW05_buf1575_R20_256 \
--name=VR_DS_MFS_sameD_toall \
--model=DSBicycle \
--batch_size=64 \
--input_nc=5 \
--output_nc=1 \
--dataset_mode=RAPIDFIM \
--save_epoch_freq=10 \
--batch=instance \
--gan_mode=wgangp \
--num_threads=8 \
--lr=4e-4 \
--niter=500 \
--niter_decay=500 
exit