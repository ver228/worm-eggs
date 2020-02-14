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


import tables
import numpy as np

import pandas as pd
import pickle

if __name__ == '__main__':
    citizens_file = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/citizen_science/consensus_events.csv'
    citizens_data = pd.read_csv(citizens_file)
    #%%
    bn = 'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4'
    predictions_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions'  / bn 
    predictions_files = {x.name[:-7] : x for x in predictions_dir.rglob('*_eggs.p')}
    
    #%%
    features_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
    ts_files = [x for x in features_dir.rglob('*.hdf5') if not (x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
    ts_files = {x.stem:x for x in ts_files}
    
    #%%
   
    def closest_index(ts, t0):
        return np.argmin(np.abs(ts - t0))
    
    for bn, annotations in citizens_data.groupby('basename'):
        if not bn in ts_files:
            continue
        
        if not bn in predictions_files:
            continue
        
        ts_file = ts_files[bn]
        with tables.File(ts_file) as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
        
        
        preds_file = predictions_files[bn]
        with open(preds_file, 'rb') as fid:
            preditions = pickle.load(fid)
        preditions = preditions['predicted_egg_flags']
        
        
        #%%
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
        #%%
#        import matplotlib.pylab as plt
#        
#        plt.figure(figsize = (15, 5))
#        plt.plot(preditions)
#        plt.plot(annotation_flags, 'o')
#        
#        #%%
#        
        break
        
        