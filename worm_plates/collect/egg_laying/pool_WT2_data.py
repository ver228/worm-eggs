#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:26:59 2020

@author: avelinojaver
"""
from pathlib import Path
import tables
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    
    src_dataset = 'v4_240pix'
    src_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/' / src_dataset
    save_dir_root = Path.home() / 'workspace/WormData/egg_laying/plates/data'
    
    
    target_shape = (96, 96)
    datasets = {}
    for dname in src_dir.glob('*/'):
        data = []
        for fname in tqdm.tqdm(dname.glob('*.hdf5')):
            with tables.File(fname, 'r') as fid:
                egg_flags = fid.get_node('/egg_flags')[:]
                snippets = fid.get_node('/snippets')[:]
            
            ss = torch.from_numpy(snippets)
            ss = F.interpolate(ss, size = target_shape)#scale_factor = 0.5), mode = 'bilinear', align_corners=False
            snippet_downsampled = ss.numpy()
            
            
            data.append((egg_flags, snippet_downsampled))
            
        
        egg_flags, snippets = map(np.concatenate, zip(*data))
        
        
        set_type = dname.stem
        save_name = save_dir_root / set_type / f'{src_dataset}_downsampled_{set_type}.hdf5'
        save_name.parent.mkdir(exist_ok = True, parents = True)
        
        filters = tables.Filters(complevel = 5, complib = 'blosc:lz4')
        with tables.File(save_name, 'w') as fid:
            fid.create_carray('/', 'snippets', obj = snippets, filters = filters)
            fid.create_carray('/', 'egg_flags', obj = egg_flags, filters = filters)