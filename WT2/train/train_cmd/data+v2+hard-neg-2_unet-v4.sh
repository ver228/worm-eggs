#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1

export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 10 \
--lr 1e-4 \
--model_name 'unet-v4' \
--data_type 'v2+hard-neg-2' \
--n_epochs 300 \
--save_frequency 50
