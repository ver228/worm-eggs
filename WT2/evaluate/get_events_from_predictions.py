#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:54:57 2019

@author: avelinojaver
"""
#%%
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tables
import pickle
import pandas as pd

def _closest_index(ts, t0):
    return np.argmin(np.abs(ts - t0))

def index2timestamps(annotations, timestamp_time):
    annotation_flags = np.full_like(timestamp_time, np.nan)
    for _, row in annotations.iterrows():
        start_point = row['start_point']
        end_point = start_point + row['video_length']
        
        
        start_ind = _closest_index(timestamp_time, start_point)
        end_ind = _closest_index(timestamp_time, end_point)
        
        
        annotation_flags[start_ind:end_ind+1] = 0
        
        events = row['timings']
        if events == events:
            events = [float(x) for x in events.split(';')]
            for e in events:
                ind = _closest_index(timestamp_time, e + start_point)
                annotation_flags[ind] = 1
    
    return annotation_flags
#%%
if __name__ == '__main__':
    threshold = 0.85
    
    #predictions_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4/'
    predictions_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10/'
    videos_dir = Path.home() / 'workspace/WormData/screenings/single_worm/finished'
    
    assert videos_dir.exists()
    assert videos_dir.exists()
    
    ts_files = [x for x in videos_dir.rglob('*.hdf5') if not (x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
    ts_files = {x.stem:x for x in ts_files}
    
    
    #%%
    
    all_rows = []
    for bn, ts_file in tqdm(ts_files.items()):
        src_file = predictions_dir / (bn + '_eggs.p')
        if not src_file.exists():
            continue
        
        with open(src_file, 'rb') as fid:
            results = pickle.load(fid)
            
        with tables.File(ts_file) as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
        total_time = timestamp_time[-1]
        
        event_frames, =  np.where(results['predicted_egg_flags'] > threshold)
        n_events = event_frames.size
        
        if event_frames.size > 0:
            
            event_times = timestamp_time[event_frames]
            
            event_frames_str = ';'.join([str(t) for t in event_frames])
            event_times_str = ';'.join([str(t) for t in event_times])
            
            
        else:
            event_frames_str = ''
            event_times_str = ''
        
        
        
        
        row = (bn, str(src_file.parent), total_time, n_events, event_frames_str, event_times_str)
        
        all_rows.append(row)
        
    
        #%%
    df = pd.DataFrame(all_rows, columns = ['basename', 'results_dir', 'total_time', 'n_events', 'events_frames', 'events_time'])
    
    
    save_dir = predictions_dir / f'egg_laying_events_th{threshold}.csv'
    df.to_csv(save_dir, index = False)
    
    