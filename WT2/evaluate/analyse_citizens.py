#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:54:11 2019

@author: avelinojaver
"""

import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt

import pandas as pd
import tables

import sys
from scipy.optimize import linear_sum_assignment

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))
ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events.csv'

def closest_index(ts, t0):
    return np.argmin(np.abs(ts - t0))

if __name__ == '__main__':
    frames_offset = 60
    
    citizens_file = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/citizen_science/consensus_events.csv'
    citizens_data = pd.read_csv(citizens_file)
    citizens_g = citizens_data.groupby('basename').groups
    
    
    df_true = pd.read_csv(ANNOTATIONS_FILE)
    df_true = df_true[df_true['set_type'] == 'test']
    
    
    old_root = '/Volumes/behavgenom_archive$'
    new_root = str(Path.home() / 'workspace/WormData/screenings')
    
    metrics = np.zeros(3)
    for bn, bn_data in tqdm(df_true.groupby('base_name')):
        true_eggs_real = np.sort(bn_data['frame_number'].values)
        
        ts_dir = bn_data['results_dir'].iloc[0].replace(old_root, new_root)
        
        ts_file = Path(ts_dir) / (bn + '.hdf5')
        if not ts_file.exists():
            continue
        
        with tables.File(ts_file) as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
        
        inds = citizens_g[bn]
        annotations = citizens_data.loc[inds]
        
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
        
        
        pred_on_real, = np.where(annotation_flags > 0.5)
        
        smooth_offset = 30
        true_eggs = np.unique(np.round(true_eggs_real/smooth_offset)*smooth_offset)
        
        pred_on = np.unique(np.round(pred_on_real/smooth_offset)*smooth_offset)
        
        
        #%%
        TP = 0
        if pred_on.size and true_eggs.size:
            dist = np.abs(true_eggs[:, None] - pred_on[None, :])
            i_true, i_preds = linear_sum_assignment(dist)
            valid_d = dist[i_true, i_preds]
            TP = (valid_d < frames_offset).sum()
            
            
                
        #%%
        FN = len(true_eggs) - TP
        
        FP = len(pred_on) - TP
        
        
        assert FP >= 0
        assert FN >= 0
        
        metrics += (TP, FP, FN)
        
#        
#        plt.figure()
#        plt.plot(annotation_flags)
#        plt.plot(pred_on, np.ones_like(pred_on), 'ob')
#        plt.plot(true_eggs, np.ones_like(true_eggs), 'xr')
#        
#        
#        plt.title((TP, FP, FN))
        
    #%%
    TP, FP, FN = metrics
            
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    
    print(P, R, F1)
    
        #%%
#    all_metrics.append(metrics)
#    
#    #%%
#    
#    
#    scores = np.zeros((len(all_metrics), len(th2check), 3))
#    
#    for im, metrics in enumerate(all_metrics):
#        TP, FP, FN = metrics.T
#        
#        P = TP/(TP+FP)
#        R = TP/(TP+FN)
#        F1 = 2*P*R/(P+R)
#        scores[im] = np.array((P, R, F1)).T
#        
#    
#    #%%
#    fig, axs = plt.subplots(1,2, figsize = (15, 5))
#    
#    
#    for ss, bn in zip(scores, models2check):
#    
#        axs[0].plot(ss[..., 0].T, ss[..., 1].T, label = bn)
#        
#    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
#    axs[0].set_xlabel('Precision')
#    axs[0].set_ylabel('Recall')
#    
#    
#    
#    axs[0].legend()
#    
#    for ss, bn in zip(scores, models2check):
#        axs[1].plot(th2check, ss[..., -1], label = bn)
#    
#    
#    
#    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
#    axs[1].set_xlabel('Thresholds')
#    axs[1].set_ylabel('F1-score')
#    #plt.legend()
#    
#    
#    for ax in axs:
#        ax.set_xlim(0, 1)
#        ax.set_ylim(0, 1)
#    
#   