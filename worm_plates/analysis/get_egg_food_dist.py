#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:07:53 2019

@author: avelinojaver
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np

if __name__ == '__main__':
    
    root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/Results'
    
    ext = '_featuresN.hdf5'
    feats_files = [x for x in root_dir.rglob('*' + ext) if not x.name.startswith('.')]
    files_dict = {x.name[:-len(ext)] : x for x in feats_files}
    #%%
    bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    events_dir = Path.home() / 'workspace/WormData/egg_laying_test/' / bn
    
    #postfix = '_eggs_full.csv'
    #save_prefix = '_dist_from_cnt_full.csv'
    
    postfix = '_eggs_events_filtered.csv'
    save_prefix = '_dist_from_cnt.csv'
    
    events_files = [x for x in events_dir.rglob('*' + postfix) if not x.name.startswith('.')]
    
    
    
    #%%
    n_missing_food_cnt = 0
    for events_file in tqdm(events_files):
        events_df = pd.read_csv(events_file)
        k = events_file.name[:-len(postfix)]
        feats_file = files_dict[k]
        
        save_file = events_file.parent / (k + save_prefix)
        
        with pd.HDFStore(feats_file, 'r') as fid:
            if not '/food_cnt_coord' in fid:
                n_missing_food_cnt += 1
                continue
            food_cnt_coord = fid.get_node('/food_cnt_coord')[:]
            microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
            
        if postfix == '_eggs_full.csv':
            events_df = events_df[events_df['frame_number'] == events_df['frame_number'].max()]
            
        egg_coords = events_df[['x', 'y']].values*microns_per_pixel
        
        dist_from_cnt = []
        for p in egg_coords:
            d = cv2.pointPolygonTest(food_cnt_coord, tuple(p), True)
            dist_from_cnt.append(d)
        dist_from_cnt = np.array(dist_from_cnt)
        
        
        dist_from_cnt = {
                'frame_number' :  events_df['frame_number'],
                'dist_from_cnt' : dist_from_cnt
                }
        
        dist_from_cnt = pd.DataFrame(dist_from_cnt)
        dist_from_cnt.to_csv(save_file)
        
        
        #%%
        
        
        
        
        