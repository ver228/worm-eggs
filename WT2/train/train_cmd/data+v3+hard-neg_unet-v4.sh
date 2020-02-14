export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 8 \
--lr 1e-4 \
--model_name 'unet-v4' \
--data_type 'v3+hard-neg' \
--resume_path $HOME/'workspace/WormData/egg_laying/single_worm/results/WT+v3_unet-v4_20191025_161425_adam-_lr0.0002_wd0.0_batch8/checkpoint-199.pth.tar' \
--n_epochs 151 \
--save_frequency 50