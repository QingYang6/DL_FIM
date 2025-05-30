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
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U10T12_S1S1LCC_O30_R50 \
--name=MUNITSDG_compare2simple \
--model=MUNITSDG \
--batch_size=6 \
--input_nc=6 \
--output_nc=2 \
--dataset_mode=S1S1LIALCCDEM \
--save_epoch_freq=1 \
--save_latest_freq=56000 \
--batch=AdaIn \
--gan_mode=wgangp \
--num_threads=8 \
--lr=4e-4 \
--niter=30 \
--niter_decay=120 
exit
#TXINCL1WEI_S1S1LIALCCDEMSLRL_U10_condi_SGD
#TXINCL1_S1S1LIALCCDEMSLRL_diversity_U10_condi_SGD
#S1S2IALCCDEM  S1S1LIALCCDEM  U10T24_S1S1LCC_O30_R300_sameOT_DEM U10T12_S1S1LCC_O30_R50 U10T24_S1S1S2_LCCDEM_O30_R200_128_QF