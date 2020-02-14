#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:07:53 2019

@author: avelinojaver
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import tables
import numpy as np

if __name__ == '__main__':
    
    root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/Results'
    
    ext = '_featuresN.hdf5'
    feats_files = [x for x in root_dir.rglob('*' + ext) if not x.name.startswith('.')]
    files_dict = {x.name[:-len(ext)] : x for x in feats_files}
    #%%
    #bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    events_dir = Path.home() / 'workspace/WormData/egg_laying_test/' / bn
    
    postfix = '_eggs_full.csv'
    events_files = [x for x in events_dir.rglob('*' + postfix) if not x.name.startswith('.')]
    
    #%%
    egg_rates = []
    
    n_missing_food_cnt = 0
    for events_file in tqdm(events_files):
        events_df = pd.read_csv(events_file)
        base_name = events_file.name[:-len(postfix)]
        feats_file = files_dict[base_name]
        
        
        
        masks_file = str(feats_file).replace('/Results/', '/MaskedVideos/').replace('_featuresN', '')
        with tables.File(masks_file, 'r') as fid:
            save_interval = fid.get_node('/full_data')._v_attrs['save_interval']
            n_full_frames = fid.get_node('/full_data').shape[0]
        
        with pd.HDFStore(feats_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
            fps = fid.get_node('/trajectories_data')._v_attrs['fps']
            microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
        
        
        counts_per_frame = np.bincount(trajectories_data['frame_number'].values)
        n_worms = np.percentile(counts_per_frame, 99)
        
        delta_t = (n_full_frames-1)*save_interval/fps/60
        
        if events_df['frame_number'].size > 0:
            eggs_per_frame = np.bincount(events_df['frame_number'].values)
            
            n_eggs_first_frame, n_eggs_last_frame = eggs_per_frame[0], eggs_per_frame[-1]
            delta_eggs = n_eggs_last_frame - n_eggs_first_frame
            delta_eggs = max(delta_eggs, 0)
            
            egg_laying_rate = delta_eggs/delta_t/n_worms
        else:
            n_eggs_first_frame = n_eggs_last_frame = egg_laying_rate = 0
        
        strain_name = base_name.partition('_')[0]
        dd = (base_name, strain_name, n_worms, n_eggs_first_frame, n_eggs_last_frame, delta_eggs, delta_t, egg_laying_rate)
        egg_rates.append(dd)
    
    save_name =  events_file.parent / 'egg_laying_rate.csv'
    columns = ['base_name', 'strain_name', 'n_worms', 'n_eggs_first_frame', 'n_eggs_last_frame', 'delta_n_eggs', 'delta_time[min]', 'egg_laying_rate[n_eggs/min/n_worms]']
    egg_rates_df = pd.DataFrame(egg_rates, columns = columns)
    egg_rates_df.to_csv(save_name)