#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:11:58 2019

@author: avelinojaver
"""
import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))
ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events_corrected.csv'


from tqdm import tqdm
import pandas as pd
import os

from process_files_all import load_model, process_files, model2gpus

def process_model(
    cuda_id = 0,
    snippet_size = 128,
    model_path_root = None,
    model_base = 'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4',
    root_data_dir = Path.home() / 'workspace/WormData/screenings/single_worm/',
    
    is_only_test = False,
    model_fname = 'model_best.pth.tar',
    bad_prefix = '/Volumes/behavgenom_archive$/single_worm/'
    ):
    
    df = pd.read_csv(ANNOTATIONS_FILE)
    if is_only_test:
        df = df[df['set_type'] == 'test']
    
    
    
    
    fnames = []
    for bn, bn_data in tqdm(df.groupby('base_name')):
        
        dname = bn_data['results_dir'].iloc[0]
        subdir = dname.replace(bad_prefix, '')
        assert subdir[0] != os.sep
        
        fname = root_data_dir / subdir / (bn + '.hdf5')
        fnames.append(fname)
    
    
    if model_path_root is None:
        model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    
    model_path_root = Path(model_path_root)
    assert model_path_root.exists()
    
    
    model_path = model_path_root / model_base / model_fname
    
    
    subdir = model_fname.partition('.')[0]
    
    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/test' / subdir / model_base 
    root_save_dir.mkdir(exist_ok = True, parents = True)
    
    model, model_args = load_model(model_path)
    model, device, batch_size = model2gpus(model, cuda_id)
    
    
    
    process_files(model, device, snippet_size, fnames, root_save_dir, model_args, batch_size = batch_size)
    
        


if __name__ == '__main__':
    cuda_id = 0
    
#    bn2check = ['WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4',
#          'WT+v2+hard-neg_unet-v1_20190823_153141_adam-_lr0.0001_wd0.0_batch4',
#            'WT+v2+hard-neg-2_unet-v3-bn_20190906_113242_adam-_lr0.001_wd0.0_batch4',
#            'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4'
#            ]
    #model_fname = 'model_best.pth.tar'
    #model_fname = f'checkpoint-{epoch2check}.pth.tar'
    
    #bn2check = ['WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4']
    #model_fname = f'checkpoint.pth.tar'
    #snippet_size = 100
    
    
#    bn2check = ['WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8']
#    model_fname = 'checkpoint-99.pth.tar'
#    snippet_size = 160
    
    
#    bn2check = ['WT+v3+hard-neg_unet-v3+R_20191029_084003_adam-_lr0.0001_wd0.0_batch4']
#    model_fname = 'checkpoint-99.pth.tar'
#    snippet_size = 100
    
    bn2check = [
                 'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4',
                 'WT+v3+hard-neg_unet-v4+R_20191028_205224_adam-_lr0.0001_wd0.0_batch8',
                 'WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8',
                 'WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4',
                 'WT+v4_unet-v4_20191029_153801_adam-_lr0.0001_wd0.0_batch10_frame-shift+kernel-simple',
                 'WT+v4_unet-v4_20191029_181002_adam-_lr0.0001_wd0.0_batch10_frame-shift',
                 'WT+v4_unet-v4_20191030_143735_adam-_lr0.0001_wd0.0_batch10_intensity+kernel-simple',
                 'WT+v4_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v2+hard-neg-2_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
                 
                 'WT+v4_unet-v4+R_20191102_175649_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
                ]
    #model_fname = 'checkpoint.pth.tar'
    model_fname = 'model_best.pth.tar'
    snippet_size = 160
    
    is_only_test = True
    #is_only_test = False
    
    for bn in bn2check:
        snippet_size = 160 if 'unet-v4' in bn else 100
        
        process_model(model_base = bn, 
                      snippet_size = snippet_size, 
                      cuda_id = cuda_id, 
                      model_fname = model_fname,
                      is_only_test = is_only_test
                      )