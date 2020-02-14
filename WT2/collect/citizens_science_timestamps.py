#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:11:58 2019

@author: avelinojaver
"""
import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))

from tqdm import tqdm
import tables
import numpy as np
import pandas as pd

if __name__ == '__main__':
    citizens_file = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/citizen_science/consensus_events.csv'
    citizens_data = pd.read_csv(citizens_file)
    
    checked_annotations = pd.read_csv('single_user_labels/egg_events.csv')
    
    citizens2check = citizens_data[~citizens_data['basename'].isin(checked_annotations['base_name'].values)]
    
    #%%
    features_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
    ts_files = [x for x in features_dir.rglob('*.hdf5') if not (x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
    ts_files = {x.stem:x for x in ts_files}
    
    #%%
   
    def closest_index(ts, t0):
        return np.argmin(np.abs(ts - t0))
    
    
    events_data = []
    for bn, annotations in tqdm(citizens2check.groupby('basename')):
        if not bn in ts_files:
            continue
        
        ts_file = ts_files[bn]
        with tables.File(ts_file) as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
        
        
        annotation_flags = np.full_like(timestamp_time, np.nan)
        for _, row in annotations.iterrows():
            start_point = row['start_point']
            end_point = start_point + row['segment_size']
            
            
            start_ind = closest_index(timestamp_time, start_point)
            end_ind = closest_index(timestamp_time, end_point)
            
            annotation_flags[start_ind:end_ind+1] = 0
            
            events = row['events']
            if events == events:
                events = [float(x) for x in events.split(';')]
                for e in events:
                    ind = closest_index(timestamp_time, e)
                    annotation_flags[ind] = 1
        
        #if there are a lot of unlabeled frames continue
        if np.isnan(annotation_flags).sum() > 1000:
            continue
        
        events, = np.where(annotation_flags>0)
        if not events.size:
            continue
        
        events_str = ' ; '.join([str(x) for x in events])
        
        row = (ts_file, events_str)
        events_data.append(row)
        
    
    df = pd.DataFrame(events_data, columns = ['path', 'events'])
    df.to_csv(Path.home() / 'events2check.csv')
    