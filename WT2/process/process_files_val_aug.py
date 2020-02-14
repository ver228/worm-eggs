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
ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events.csv'


from train.models import get_model
from cell_localization.utils import  get_device

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import tables
import pickle
import pandas as pd
import numpy as np
import os

@torch.no_grad()
def _get_file_predictions(fname, batch_size):
    
    gen = FlowFile(fname, batch_size)
    loader = DataLoader(gen, 
                        batch_size=1, 
                        num_workers=1
                        )
    
    
    egg_flags = np.full((4, gen.tot_frames), np.nan)
    for t, X_ori in tqdm(loader):
        assert X_ori.shape[-1] == X_ori.shape[-2]
        
        X_ori = X_ori.to(device)
        for ii in range(4):
            
            if ii == 0:
                X = X_ori
            elif ii == 1:
                X = X_ori.flip(-1)
            elif ii == 1:
                X = X_ori.flip(-2)
            else:
                X = X_ori.transpose(-1, -2)
                
            prediction = model(X)
            prediction = prediction[0].squeeze(-1)
            prediction = torch.sigmoid(prediction)
            prediction = prediction.detach().cpu().numpy()
            
            #I want to remove the corners
            prediction = prediction[gen.offset:- gen.offset]
            
            t0 = t + gen.offset
            egg_flags[ii, t0:t0 + prediction.size] = prediction
    
    gen.fid.close()
    
        
    return egg_flags


class FlowFile():
    #this should only work with one process at the time otherwise it might cause collitions when reading...
    offset = 3
    valid_size = [(150,-150), (230,-230)]
    def __init__(self, fname, batch_size):
        self.fid = tables.File(fname, 'r')
        self.masks = self.fid.get_node('/mask') 
        self.tot_frames = self.masks.shape[0]
        self.times2read = [(x, x +batch_size) for x in  range(0, self.tot_frames, batch_size - 2*self.offset)]
    
    def __getitem__(self, ind):
        t1, t2 = self.times2read[ind]
        (xl, xr), (yl, yr) = self.valid_size
        imgs = self.masks[t1:t2, xl:xr, yl:yr]
        X = torch.from_numpy(imgs)
        X = X.float()/255
        return t1, X
    
    def __len__(self):
        return len(self.times2read)
        
if __name__ == '__main__':
    cuda_id = 0
    batch_size = 100
    device = get_device(cuda_id)
    
    bn2check = ['WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4',
          'WT+v2+hard-neg_unet-v1_20190823_153141_adam-_lr0.0001_wd0.0_batch4',
            'WT+v2+hard-neg-2_unet-v3-bn_20190906_113242_adam-_lr0.001_wd0.0_batch4',
            'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4'
            ]
    
    
    epoch2check = 199
    for bn in bn2check:
        #model_path = Path.home() / 'workspace/WormData/egg_laying/results/' / bn / 'model_best.pth.tar'
        model_path = Path.home() / 'workspace/WormData/egg_laying/results/' / bn / f'checkpoint-{epoch2check}.pth.tar'
        
        root_save_dir = Path.home() / 'workspace/WormData/egg_laying/predictions/' / f'aug-{epoch2check}' / bn
        
        
        bad_prefix = '/Volumes/behavgenom_archive$/single_worm/'
        root_data_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
        
        
        
        state = torch.load(model_path, map_location = 'cpu')
        state_dict = {}
        for k,v in state['state_dict'].items():
            if 'output_block.' in k:
                continue
            if '.net.' not in k and k.startswith('mapping_network.'):
                k = k[:16] + 'net.' + k[16:]
            state_dict[k] = v
        
        model_name = bn.split('_')[1]
        if model_name.endswith('+pretrained'):
            model_name = model_name.replace('+pretrained', '')
        
        model = get_model(model_name, n_in = 1, n_out = 1)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        df = pd.read_csv(ANNOTATIONS_FILE)
        df = df[df['set_type'] == 'test']
        
        
        #I am making the assumption that `bn` is unique for each video
        for bn, bn_data in tqdm(df.groupby('base_name'), desc = bn):
            true_eggs = bn_data['frame_number'].values
            
            dname = bn_data['results_dir'].iloc[0]
            subdir = dname.replace(bad_prefix, '')
            assert subdir[0] != os.sep
            
            fname = root_data_dir / subdir / (bn + '.hdf5')
            if not fname.exists():
                continue
            
            save_name = root_save_dir / bn_data.iloc[0]['set_type'] / (bn + '_eggs.p')
            save_name.parent.mkdir(exist_ok = True, parents = True)
            
            with tables.File(fname, 'r') as fid:
                stage_position_pix = fid.get_node('/stage_position_pix')[:]
                is_stage_move = np.isnan(stage_position_pix[:, 0])
            
            
            egg_flags = _get_file_predictions(fname, batch_size)
            data = dict(
                    batch_size = batch_size,
                    model_path = str(model_path),
                    model_epoch = state['epoch'],
                    predicted_egg_flags = egg_flags,
                    is_stage_move = is_stage_move,
                    true_eggs = true_eggs,
                    video_file = str(fname)
                    )
            
            with open(save_name, 'wb') as fid:
                pickle.dump(data, fid)
        
        
    #    train_loader = DataLoader(train_flow, 
    #                            batch_size=batch_size, 
    #                            shuffle=True, 
    #                            num_workers=num_workers
    #                            )