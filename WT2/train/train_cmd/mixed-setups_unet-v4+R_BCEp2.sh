export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 32 \
--lr 5e-5 \
--model_name 'unet-v4' \
--loss_name 'BCEp2' \
--data_type 'mixed-setups' \
--save_frequency 200 \
--resume_path $HOME/'workspace/WormData/egg_laying/single_worm/results/WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8/checkpoint-99.pth.tar' \
--n_epochs 1001 