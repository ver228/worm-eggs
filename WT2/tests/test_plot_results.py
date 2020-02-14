#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:31:20 2019

@author: avelinojaver
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
if __name__ == '__main__':
    
    
    #src_dir = '/Volumes/rescomp1/data/WormData/egg_laying/predictions/WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4/test/'
    #src_dir = '/Volumes/rescomp1/data/WormData/egg_laying/predictions/WT+v2+hard-neg_unet-v1_20190823_153141_adam-_lr0.0001_wd0.0_batch4/test/'
    #src_dir = '/Volumes/rescomp1/data/WormData/egg_laying/predictions/WT+v2+hard-neg-2_unet-v3-bn_20190906_113242_adam-_lr0.001_wd0.0_batch4/test'
    src_dir = '/Volumes/rescomp1/data/WormData/egg_laying/predictions/WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4/test'
    
    for src_file in Path(src_dir).glob('*.p'):
        with open(src_file, 'rb') as fid:
            results = pickle.load(fid)
            
        #%%
        predictions = results['predicted_egg_flags']
        true_eggs = results['true_eggs']
        
        #%%
        pred_on, = np.where(predictions>0.6)
        
        
        valid_with_offset = np.concatenate((true_eggs - 1, true_eggs, true_eggs+1))
        wrong_events = set(pred_on) - set(valid_with_offset)
        
        pred_with_offset = np.concatenate((pred_on - 1, pred_on, pred_on+1))
        valid_events = set(true_eggs) & set(pred_with_offset)
        missing_events = set(true_eggs) - valid_events
        
        
        valid_events = list(valid_events)
        wrong_events = list(wrong_events)
        missing_events = list(missing_events)
        
        plt.figure(figsize = (20, 5))
        plt.plot(predictions)
        plt.plot(valid_events, predictions[valid_events], 'go')
        plt.plot(missing_events, predictions[missing_events], 'rs')
        plt.plot(wrong_events, predictions[wrong_events], 'rx')
        plt.title(src_file.stem)
        plt.ylim([-0.05, 1.05])
        
        
        
        #%%
#        snippet_size = 11
#        snippet_offset = 3
#        min_separation = 5
#        max_snippets = 10
#        
#        wrong_events = sorted(wrong_events)
#        selected_events = [wrong_events[0]]
#        for w in wrong_events[1:]:
#            if w - selected_events[-1] >= min_separation:
#                selected_events.append(w)
#        selected_events = sorted(selected_events, key = lambda x : predictions[x], reverse = True)
#        selected_events = selected_events[:max_snippets]
#        
#        tot_frames = predictions.shape[0]
#        windows2read = []
#        for ev in selected_events:
#            l = max(0, ev - snippet_offset)
#            if l + snippet_size > tot_frames:
#                l = tot_frames - snippet_size
#            windows2read.append((l, l+snippet_size))
        
        
        
        
        