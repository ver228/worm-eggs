export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/worm-eggs/WT2/train

python train.py \
--batch_size 4 \
--lr 1e-4 \
--model_name 'unet-v3' \
--data_type 'v3+hard-neg' \
--resume_path '$HOME/workspace/WormData/egg_laying/single_worm/results/WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4/checkpoint.pth.tar' \
--n_epochs 100
