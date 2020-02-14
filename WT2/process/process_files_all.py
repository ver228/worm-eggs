#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:11:58 2019

@author: avelinojaver
"""
import sys
import os
from pathlib import Path
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))

from train.models import get_model
from cell_localization.utils import  get_device

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import tables
import pickle
import numpy as np
import random

dname = Path(__file__).resolve().parents[1]
TEST_FILES = dname / 'collect/single_user_labels/egg_events_231019_files.txt'

@torch.no_grad()
def get_file_predictions(model, device, fname, snippet_size, batch_size, desc_tqdm = ''):
    
    
    gen = FlowFile(fname, snippet_size)
    
#    if is_distributed:
#        sampler = torch.utils.data.distributed.DistributedSampler(gen)
#    else:
#        sampler = None
    
    
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        num_workers = 1,
                        #shuffle=(sampler is None),
                        #sampler = sampler,
                        pin_memory = True
                        )
    
    
    egg_flags = np.full(gen.tot_frames, np.nan)
    for ini_times, X in tqdm(loader, desc = desc_tqdm):
        X = X.to(device)
            
        predictions = model(X)
        predictions = torch.sigmoid(predictions)
        predictions = predictions.squeeze(-1)
        predictions = predictions.detach().cpu().numpy()
        
        for t, prediction in zip(ini_times, predictions):
        
            #I want to remove the corners
            prediction = prediction[ gen.offset:- gen.offset]
            
            t0 = t + gen.offset
            egg_flags[t0:t0 + prediction.size] = prediction
    
    gen.fid.close()
    
        
    return egg_flags


class FlowFile():
    #this should only work with one process at the time otherwise it might cause collitions when reading...
    offset = 3
    valid_size = [(150,-150), (230,-230)]
    def __init__(self, fname, snippet_size):
        self.snippet_size  = snippet_size
        
        self.fid = tables.File(fname, 'r')
        self.masks = self.fid.get_node('/mask') 
        self.tot_frames = self.masks.shape[0]
        
        
        times2read = list(range(0, self.tot_frames, snippet_size - 2*self.offset))
        #i want to force all the times to have the same batch size
        times2read = [min(x, self.tot_frames - snippet_size) for x in times2read] 
        self.times2read = [(x, x + snippet_size)  for x in times2read]
    
    
    def __getitem__(self, ind):
        t1, t2 = self.times2read[ind]
        (xl, xr), (yl, yr) = self.valid_size
        imgs = self.masks[t1:t2, xl:xr, yl:yr]
        
        X = torch.from_numpy(imgs)
        X = X.float()/255
        return t1, X
    
    def __len__(self):
        return len(self.times2read)

def load_model(model_path):
    state = torch.load(model_path, map_location = 'cpu')
    state_dict = {}
    for k,v in state['state_dict'].items():
        if 'output_block.' in k:
            continue
        if '.net.' not in k and k.startswith('mapping_network.'):
            k = k[:16] + 'net.' + k[16:]
        state_dict[k] = v
    
    
    bn = model_path.parent.name
    
    model_name = bn.split('_')[1]
    model_name = model_name.replace('+hardaug', '') #I do not need this prefix. It is only to indicate it was trained with a different augmentation parameters
    
    model = get_model(model_name, n_in = 1, n_out = 1)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    model_args = dict(
            model_path = str(model_path),
            model_epoch = state['epoch']
            )
    
    
    
    return model, model_args


def model2gpus(model, cuda_id):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  
    batch_size = max(1, torch.cuda.device_count())
    
    device = get_device(cuda_id)
    model = model.to(device)
    
    return model, device, batch_size

def process_files(model, device,  snippet_size, fnames, root_save_dir, model_args, batch_size = 1):
    
    random.shuffle(fnames)
    for ifname, fname in enumerate(tqdm(fnames)):
        if not fname.exists():
            continue
        
        bn = fname.stem
        save_name = root_save_dir / (bn + '_eggs.p')
        if save_name.exists():
            continue
        
        try:
            with tables.File(fname, 'r') as fid:
                fid.get_node('/mask')
        except:
            continue
            
        with tables.File(fname, 'r') as fid:
            stage_position_pix = fid.get_node('/stage_position_pix')[:]
            is_stage_move = np.isnan(stage_position_pix[:, 0])
            timestamp_time = fid.get_node('/timestamp/time')[:]
        
        egg_flags = get_file_predictions(model, device, fname, snippet_size, batch_size, desc_tqdm = f'Files:{ifname}/{len(fnames)}')
        
        data = dict(
                snippet_size = snippet_size,
                batch_size = batch_size,
                predicted_egg_flags = egg_flags,
                is_stage_move = is_stage_move,
                timestamp_time = timestamp_time,
                video_file = str(fname)
                )
        data.update(model_args)
        with open(save_name, 'wb') as fid:
            pickle.dump(data, fid)
        

def process_files_all(
    cuda_id = 0,
    snippet_size = 160,#128,
    model_path_root = None,
    model_base = 'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4', #'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10',#
    root_data_dir = Path.home() / 'workspace/WormData/screenings/single_worm/',
    is_only_test = False
    ):
    
    if is_only_test:
        with open(TEST_FILES, 'r') as fid:
            ss = fid.read()
            fnames = [Path(os.path.expandvars(x)) for x in ss.split('\n')]
            
    else:
        fnames = root_data_dir.rglob('*.hdf5')
        fnames = [x for x in fnames if not (x.name.startswith('.') or x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
    
    random.shuffle(fnames)
    
    if model_path_root is None:
        model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    
    model_path_root = Path(model_path_root)
    assert model_path_root.exists()
    
    
    
    
    model_path = model_path_root / model_base / 'model_best.pth.tar'
    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/predictions/' / 'single_worm'  / model_base 
    root_save_dir.mkdir(exist_ok = True, parents = True)
    
    model, model_args = load_model(model_path)
    model, device, batch_size = model2gpus(model, cuda_id)
    
    process_files(model, device, snippet_size, fnames, root_save_dir, model_args, batch_size = batch_size)
        
        
if __name__ == '__main__':
    import fire
    fire.Fire(process_files_all)