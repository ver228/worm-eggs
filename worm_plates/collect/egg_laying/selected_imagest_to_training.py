#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:51:22 2020

@author: avelinojaver
"""

from pathlib import Path
import pickle
import numpy as np
import tqdm
import math
import random
import tables

if __name__ == '__main__':
    save_dir_root = Path.home() / 'workspace/WormData/egg_laying/plates/data'
    #save_dir_root = Path('/Users/avelinojaver/Desktop/syngenta_sequences/final') 
    
    sets_limits = {'test' : (0., 0.05), 'validation' : (0.05, 0.1), 'train' : (0.1, 1.)}
    
    #bn = 'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
    #bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    bn = 'WT+mixed-setups_unet-v4+R_BCE_20200207_181629_adam-_lr5e-05_wd0.0_batch32'
    
    src_dataset = 'Syngenta_' + bn
    
    src_dir = Path('/Users/avelinojaver/Desktop/syngenta_sequences/') / bn
    
    
    
    true_files = [x.stem for x in (src_dir / 'images').glob('*.png')]
    snippet_files = list((src_dir / 'snippets').glob('*.pickle'))
    
    ss = set(true_files) - set([x.stem for x in snippet_files])
    assert len(ss) == 0
    
    
    target_shape = (96, 96)
    target_length = 7
    
    def _get_lims(original_size, target_size):
        if original_size < target_size:
            return
        offset = original_size - target_size
        l = math.ceil(offset/2)
        r = original_size - math.floor(offset/2)
        return l,r
            
    
    
    data = []
    for snippet_file in tqdm.tqdm(snippet_files):
        with open(snippet_file, 'rb') as fid:
            snippet = pickle.load(fid)
        
        H, W = snippet.shape[-2:]
        if snippet.shape[0] != target_length:
            continue
        
        if H < target_shape[0] or W < target_shape[1]:
            continue
        
        
        
        lh, rh = _get_lims(H, target_shape[0])
        lw, rw = _get_lims(W, target_shape[1])
        
        snippet = snippet[:, lh:rh, lw:rw]
        assert snippet.shape[-2:] == target_shape
        
        
        
        n_images = len(snippet)
        mid = n_images//2
        egg_flag = np.zeros(n_images, np.int32)
        
        if snippet_file.stem in true_files:
            egg_flag[mid] = 1
        data.append((egg_flag, snippet))
    
    
    #%%
    random.shuffle(data)
        
    tot = len(data)
    for set_type, (frac_l, frac_r) in sets_limits.items():
        ind1 = math.ceil(frac_l*tot)
        ind2 = math.floor(frac_r*tot)
        
        set_data = data[ind1:ind2+1]
        egg_flags, snippets = map(np.stack, zip(*set_data))
        
        save_name = save_dir_root / set_type / f'{src_dataset}_downsampled_{set_type}.hdf5'
        save_name.parent.mkdir(exist_ok = True, parents = True)
        
        filters = tables.Filters(complevel = 5, complib = 'blosc:lz4')
        with tables.File(save_name, 'w') as fid:
            fid.create_carray('/', 'snippets', obj = snippets, filters = filters)
            fid.create_carray('/', 'egg_flags', obj = egg_flags, filters = filters)
        
        
        
    