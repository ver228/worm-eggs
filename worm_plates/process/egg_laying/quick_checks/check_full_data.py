#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:39:07 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import tables

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    mask_file = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos/CeNDR_Set1_020617/N2_worms10_food1-10_Set3_Pos4_Ch3_02062017_123419.hdf5'
    eggs_file = Path.home() / 'workspace/WormData/egg_laying_test/worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64/N2_worms10_food1-10_Set3_Pos4_Ch3_02062017_123419_eggs_full.csv'
    
    eggs_full_frames = pd.read_csv(eggs_file)
    
    with tables.File(mask_file) as fid:
        imgs = fid.get_node('/full_data')[:]
        
    
    for frame_number, frame_data in eggs_full_frames.groupby('frame_number'):
        img = imgs[frame_number]
        
        plt.figure(figsize = (10, 10))
        plt.imshow(img, cmap='gray')
        plt.plot(frame_data['x'], frame_data['y'], '.r')