export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 32 \
--lr 5e-5 \
--model_name 'unet-v4' \
--data_type 'mixed-setups-v2' \
--save_frequency 500 \
--resume_path $HOME/'workspace/WormData/egg_laying/single_worm/results/WT+mixed-setups_unet-v4+R_BCE_20200207_181629_adam-_lr5e-05_wd0.0_batch32/model_best.pth.tar' \
--n_epochs 1001 