#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:59:49 2019

@author: avelinojaver
"""
import pandas as pd
import numpy as np
import tables
from tqdm import tqdm
from pathlib import Path

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

if __name__ == '__main__':
    #%%
    _debug = False
    fname_citizens = '/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/2015-10-25_worms_classifications.csv'
    
    df_tierpsy = pd.read_csv('single_user_labels/egg_events.csv')
    df_tierpsy_g = df_tierpsy.groupby('base_name').groups
    
    
    features_dir = Path.home() / 'workspace/WormData/screenings/single_worm'
    old_features_dir = '/Volumes/behavgenom_archive$/single_worm'
    
    
    #%%
    df_citizens = pd.read_csv(fname_citizens)
    df_citizens = df_citizens[df_citizens['user_name'] == 'aexbrown']
    df_citizens = df_citizens.drop_duplicates()
    
    df_citizens = df_citizens[~df_citizens['file_name'].isnull()]
    df_citizens_g = df_citizens.groupby('file_name')
    #%%
    #, 'start_point']).groups
    
    
    
    
    all_flags = []
    for fname, citizens_dat in tqdm(df_citizens_g):
        bn = fname[:-len('_seg.avi')]
        #if bn != 'C52B9.11 (gk596)V on food R_2011_09_13__11_37_28___1___2':
        #    continue
        
        if not bn in df_tierpsy_g:
            continue
        
        tierpsy_dat = df_tierpsy.loc[df_tierpsy_g[bn]]
        results_dir = tierpsy_dat['results_dir'].iloc[0]
        results_dir = results_dir.replace(old_features_dir, str(features_dir))
        ts_file = Path(results_dir) / (bn + '.hdf5')
        
        if not ts_file.exists():
            continue
        
        
        with tables.File(ts_file) as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
        citizens_flags = index2timestamps(citizens_dat, timestamp_time)
        
        citizens_events, = np.where(citizens_flags>0)
        
        
        tierpsy_events = tierpsy_dat['frame_number'].values
        tierpsy_events = tierpsy_events[tierpsy_events<timestamp_time.size]
        tierpsy_flags = np.zeros_like(timestamp_time)
        tierpsy_flags[tierpsy_events] = 1
        tierpsy_flags[np.isnan(citizens_flags)] = np.nan
        tierpsy_events, = np.where(tierpsy_flags>0)
        
        all_flags.append((bn, citizens_events, tierpsy_events))
    
    #%%
    
    offsets = []
    for bn, citizens_events, tierpsy_events in all_flags:
        if citizens_events.size > 0 or citizens_events.size > 0:
            offset = np.abs(citizens_events[:, None] - tierpsy_events[None, :])
            offsets.append(offset.min(axis=0))
            #print(bn, citizens_events, tierpsy_events)
    
        
    #%%