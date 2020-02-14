export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 10 \
--lr 1e-4 \
--model_name 'unet-v4' \
--data_type 'v4' \
--resume_path $HOME/'workspace/WormData/egg_laying/single_worm/results/WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8/checkpoint-99.pth.tar' \
--n_epochs 151 \
--save_frequency 50
