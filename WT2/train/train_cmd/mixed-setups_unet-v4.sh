export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 32 \
--lr 1e-4 \
--model_name 'unet-v4' \
--data_type 'mixed-setups' \
--n_epochs 2001 \
--save_frequency 500
