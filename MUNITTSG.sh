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
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U10T24_S1S1S2_LCCDEM_O30_R200_128_QF \
--name=S1S2IALCCDEM_U10_condi_TD \
--model=MUNITTSG \
--batch_size=16 \
--input_nc=6 \
--output_nc=2 \
--dataset_mode=S1S2IALCCDEM \
--save_epoch_freq=5 \
--batch=AdaIn \
--gan_mode=wgangp \
--num_threads=8 \
--lr=4e-4 \
--niter=200 \
--niter_decay=300 
exit