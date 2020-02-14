export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 8 \
--lr 5e-2 \
--weight_decay 1e-4 \
--model_name 'unet-v4' \
--data_type 'v3' \
--lr_scheduler_name 'cosineLR-200' \
--n_epochs 210 \
--warmup_epochs 10 \
--optimizer_name 'sgd'
