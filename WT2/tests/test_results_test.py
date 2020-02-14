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

from tqdm import tqdm
from cell_localization.utils import  get_device
import torch
import tables
from train.models import get_model
import pickle
import numpy as np

if __name__ == '__main__':
    cuda_id = 0
    device = get_device(cuda_id)
    
    #fname = Path.home() / 'workspace/WormData/screenings/single_worm/finished/mutants/egl-14(n549)X@MT1179/food_OP50/XX/30m_wait/clockwise/egl-14 (n549)X on food L_2010_07_23__14_56_45___8___10.hdf5'
    fname = Path.home() / 'workspace/WormData/screenings/single_worm/finished/mutants/C11D2.2(ok1565)IV@RB1380/food_OP50/XX/30m_wait/anticlockwise/CIID2.2 (ok1565)IV on food R_2011_08_04__12_26_51___1___6.hdf5'
    #fname = Path.home() / 'workspace/WormData/screenings/single_worm/finished/mutants/acr-14(ok1155)II@RB1132/food_OP50/XX/30m_wait/anticlockwise/acr-14 (ok1155) on food R_2010_02_24__09_43___3___2.hdf5'
    
    #model_path = Path.home() / 'workspace/WormData/egg_laying/results/WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4/model_best.pth.tar'
    #model_path = Path.home() / 'workspace/WormData/egg_laying/results/WT+v2+hard-neg_unet-v1_20190821_185645_adam-_lr1e-05_wd0.0_batch4/model_best.pth.tar'
    model_path = Path.home() / 'workspace/WormData/egg_laying/results/WT+v2+hard-neg_unet-v1_20190823_153141_adam-_lr0.0001_wd0.0_batch4/model_best.pth.tar'
    
    
    save_name = Path.home() / 'workspace' / (fname.stem + '_eggs.p')
    #%%
    valid_size = [(150,-150), (230,-230)]
    (xl, xr), (yl, yr) = valid_size
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    state_dict = {}
    for k,v in state['state_dict'].items():
        if 'output_block.' in k:
            continue
        if '.net.' not in k and k.startswith('mapping_network.'):
            k = k[:16] + 'net.' + k[16:]
        state_dict[k] = v
        
    model = get_model('unet-v1', n_in = 1, n_out = 1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    #%%
    model.eval()
    
    batch_size = 96
    offset = 3
    
    
    with torch.no_grad():
        with tables.File(fname, 'r') as fid:
            masks = fid.get_node('/mask')
            
            egg_flags = np.full(masks.shape[0], np.nan)
            for t in tqdm(range(0, masks.shape[0], batch_size - 2*offset)):
                
                
                imgs = masks[t:t+batch_size, xl:xr, yl:yr]
                
                X = torch.from_numpy(imgs[None])
                X = X.float()/255
                X = X.to(device)
                
                
                prediction = model(X)
                prediction = prediction[0].squeeze(-1)
                prediction = torch.sigmoid(prediction)
                prediction = prediction.detach().cpu().numpy()
                
                #I want to remove the corners
                prediction = prediction[ offset:-offset]
                
                t0 = t + offset
                egg_flags[t0:t0 + prediction.size] = prediction
        
    
    data = dict(
            offset = offset,
            batch_size = batch_size,
            valid_size = valid_size,
            model_path = model_path,
            egg_flags = egg_flags
            )
            
    
    with open(save_name, 'wb') as fid:
        pickle.dump(data, fid)
    