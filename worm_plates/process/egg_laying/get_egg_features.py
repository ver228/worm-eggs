#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:37:37 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    #bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    
    #features_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/Results'
    #events_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/CeNDR' / bn
    
    features_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam'
    events_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/syngenta' / bn
    
    #%%
    ext = '_featuresN.hdf5'
    feats_files = [x for x in features_dir.rglob('*' + ext) if not x.name.startswith('.')]
    files_dict = {x.name[:-len(ext)] : x for x in feats_files}
    events_f_files = [x for x in events_dir.rglob('*_eggs_events_filtered.csv') if  not x.name.startswith('.')]
    
    events_f_files = {x.name[:-25]: x for x in events_f_files}
    
    #%%
    files_data = []
    for k in events_f_files.keys():
        strain, n_worms, *_ = k.split('_')
        try:
            n_worms = int(n_worms[5:])
        except:
            n_worms = np.nan
        files_data.append((k, strain, n_worms))
    
    files_data = pd.DataFrame(files_data, columns = ['basename', 'strain', 'n_worms'])
    files_data = files_data.sort_values(by = ['strain', 'n_worms']).reset_index(drop = True)
    files_data.to_pickle(events_dir / 'files_data.pkl')
    
    for ifile, file_info in tqdm(files_data.iterrows(), total = len(files_data)):
        bn = file_info['basename']
        
        egg_events_file = events_f_files[bn]
        egg_events = pd.read_csv(egg_events_file)
        
        if not len(egg_events):
            continue
        
        egg_events = egg_events.drop_duplicates()
        
        feats_file = files_dict[bn]
        with pd.HDFStore(feats_file, 'r') as fid:
            blob_features = fid['/blob_features']
            timeseries_data = fid['/timeseries_data']
        
        v_worm_data = []
        for irow, egg in egg_events.iterrows():
            worm_data = timeseries_data[timeseries_data['worm_index'] == egg['worm_index']].copy()
            
            #for this data i am assuming timestamp and frame number are the same. This should hold in data taken with Andre's worm rig
            worm_data['timestamp_centered'] = worm_data['timestamp'] - int(egg['frame_number'])
            worm_data['video_id'] = ifile
            v_worm_data.append(worm_data)
        assert len(v_worm_data) == len(egg_events)
        
        v_worm_data = pd.concat(v_worm_data, ignore_index = True)
        
        v_worm_data.to_pickle(events_dir / f'{bn}_timeseries.pkl')
    
    
    #%%
    