export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 10 \
--lr 5e-2 \
--weight_decay 1e-4 \
--model_name 'unet-v4' \
--data_type 'v4' \
--lr_scheduler_name 'cosineLR-1000' \
--warmup_epochs 10 \
--optimizer_name 'sgd' \
--n_epochs 1010 \
--save_frequency 50
