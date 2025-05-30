#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/train_novisdom.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Downscaling/VIIRs_RAPID/Samples/RAPIDonly_PerW05_buf1575_R20 \
--name=VR_Ronly_condi_sameD_CVAE_noWOP_twoDEM \
--model=CycleED \
--batch_size=16 \
--input_nc=2 \
--output_nc=1 \
--dataset_mode=RAPIDFIM \
--save_epoch_freq=1 \
--batch=instance \
--gan_mode=vanilla \
--num_threads=8 \
--lr=4e-4 \
--niter=100 \
--niter_decay=200 \
--lambda_GAN2=0 
exit