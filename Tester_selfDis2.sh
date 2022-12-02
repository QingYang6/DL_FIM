#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/Test_sulfDistribution.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U40T36_S1S1LCC_O30_R300_sameOT_DEM_FandDtest \
--name=S1S1LIALCCDEMSLRL_U10_NoC2E_SGD_temp \
--model=MUNITSDGCRL \
--batch_size=6 \
--input_nc=6 \
--output_nc=2 \
--dataset_mode=S1S1LIALCCDEM \
--save_epoch_freq=1 \
--batch=AdaIn \
--gan_mode=wgangp \
--num_threads=8 \
--lr=4e-4 \
--niter=50 \
--niter_decay=200 \
--predict_freq=10000 \
--load_iter=2464116 \
--valid_batchs=1
exit
