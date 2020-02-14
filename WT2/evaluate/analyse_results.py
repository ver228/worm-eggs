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

import sys

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))
ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events.csv'

if __name__ == '__main__':
    
    
    root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/'
    
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/_old/199/'
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/_old/aug-199/'

    
    models2check = sorted([x.name for x in root_dir.glob('*/')])
    
    all_metrics = []
    
    
    th2check = np.arange(0.0, 1, 0.025)
    
    
    df_true = pd.read_csv(ANNOTATIONS_FILE)
    df_true = df_true[df_true['set_type'] == 'test']
    
    for model_name in models2check:
        
        src_dir = root_dir / model_name
        if (src_dir / 'test').is_dir():
            src_dir = src_dir / 'test'
        
        #src_files = list(Path(src_dir).glob('*.p'))
        
        metrics = np.zeros((len(th2check), 3))
        
        for bn, bn_data in tqdm(df_true.groupby('base_name'), desc = model_name):
            true_eggs = np.sort(bn_data['frame_number'].values)
            
            src_file = src_dir / (bn + '_eggs.p')
            if not src_file.exists():
                continue
            
            with open(src_file, 'rb') as fid:
                results = pickle.load(fid)
            
            
            predictions = results['predicted_egg_flags']
            
            if predictions.ndim == 2:
                predictions = np.mean(predictions, axis=0)
            
            valid_with_offset = np.concatenate((true_eggs - 1, true_eggs, true_eggs+1))
            valid_with_offset = set(valid_with_offset)
            
            #%%
            
            for ith, th in enumerate(th2check):
                pred_on, = np.where(predictions > th)
                
                wrong_events = set(pred_on) - valid_with_offset
                
                pred_with_offset = np.concatenate((pred_on - 1, pred_on, pred_on+1))
                valid_events = set(true_eggs) & set(pred_with_offset)
                missing_events = set(true_eggs) - valid_events
                
                
                valid_events = list(valid_events)
                wrong_events = list(wrong_events)
                missing_events = list(missing_events)
                
                
                TP = len(valid_events)
                FP = len(wrong_events)
                FN = len(missing_events)
                metrics[ith] += (TP, FP, FN)
            #%%
        all_metrics.append(metrics)
    
    #%%
    
    scores = np.zeros((len(all_metrics), len(th2check), 3))
    for im, metrics in enumerate(all_metrics):
        TP, FP, FN = metrics.T
        
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        scores[im] = np.array((P, R, F1)).T
        
    #%%
    fig, axs = plt.subplots(1,2, figsize = (15, 5))
    
    
    for ss, bn in zip(scores, models2check):
    
        axs[0].plot(ss[..., 0].T, ss[..., 1].T, label = bn)
        
    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
    axs[0].set_xlabel('Precision')
    axs[0].set_ylabel('Recall')
    
    
    
    axs[0].legend()
    
    for ss, bn in zip(scores, models2check):
        axs[1].plot(th2check, ss[..., -1], label = bn)
    
    
    
    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
    axs[1].set_xlabel('Thresholds')
    axs[1].set_ylabel('F1-score')
    #plt.legend()
    
    
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
   