#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:41:33 2019

@author: avelinojaver
"""

from pathlib import Path
from tqdm import tqdm
import tables
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    src_dir = Path.home() / 'workspace/localization/data/worm_eggs_with_masks/'
    fnames = [x for x in src_dir.rglob('*eggs_mask.hdf5') if not x.name.startswith('.')]
    
    
    imgs_metrics = []
    
    for fname in tqdm(fnames):
        with tables.File(str(fname), 'r') as fid:
            
            img = fid.get_node('/img')[:]
            
            valid_pix = img[img>0]
            img_mean = np.mean(valid_pix)
            img_std = np.std(valid_pix)
            img_max = np.max(valid_pix)
            img_min = np.min(valid_pix)
            
            
            imgs_metrics.append((img_mean, img_std, img_max, img_min))
            
    img_mean, img_std, img_max, img_min = map(np.array, zip(*imgs_metrics))
    #%%
    for rr in [img_mean, img_std, img_max, img_min]:
        print((rr.min(), rr.mean(), rr.max()))