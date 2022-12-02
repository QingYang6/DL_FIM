#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
lsb_release -a
echo "SLURM_JOBID ${SLURM_JOBID} SLURM_JOBID"
source /home/qiy17007/miniconda3/bin/activate torch3
python3 -u /shared/stormcenter/Qing_Y/GAN_ChangeDetection/ASGIT/train_novisdom.py \
--dataroot=/gpfs/sharedfs1/manoslab/data/Qdata/GAN_CD/Sample_Preare/Huston/U10T12_S1S1LCC_O30_R50 \
--name=S1S1LIALCC_uccycle_CtoED_U10_sameGD \
--model=MUNITUCCLE \
--batch_size=5 \
--input_nc=5 \
--output_nc=2 \
--dataset_mode=S1S1LIALCC \
--save_epoch_freq=1 \
--batch=AdaIn \
--gan_mode=wgangp \
--num_threads=8 \
--lr=4e-4 \
--niter=30 \
--niter_decay=170 \
--predict_freq=10000
exit