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
    
    #src_dir = '/Volumes/rescomp1/data/WormData/egg_laying/predictions/WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4/train/'
    #for src_file in Path(src_dir).glob('*15_49_17*.p'):
    
    #bn = 'egl-14 (n549)X on food L_2010_07_23__14_56_45___8___10_eggs.p'
    bn = 'CIID2.2 (ok1565)IV on food R_2011_08_04__12_26_51___1___6_eggs.p'
    #bn = 'acr-14 (ok1155) on food R_2010_02_24__09_43___3___2_eggs.p'
    src_file = Path.home() / 'workspace' / bn
    
    
    with open(src_file, 'rb') as fid:
        results = pickle.load(fid)
            
        #%%
    egg_flags = results['egg_flags']
    
    #%%
    pred_on, = np.where(egg_flags>0.6)
    
    plt.figure(figsize = (20, 5))
    plt.plot(egg_flags)
#    plt.plot(valid_events, predictions[valid_events], 'go')
#    plt.plot(missing_events, predictions[missing_events], 'rs')
#    plt.plot(wrong_events, predictions[wrong_events], 'rx')
#    plt.title(src_file.stem)
#    
    #%%
#    snippet_size = 11
#    snippet_offset = 3
#    min_separation = 5
#    max_snippets = 10
#    
#    wrong_events = sorted(wrong_events)
#    selected_events = [wrong_events[0]]
#    for w in wrong_events[1:]:
#        if w - selected_events[-1] >= min_separation:
#            selected_events.append(w)
#    selected_events = sorted(selected_events, key = lambda x : predictions[x], reverse = True)
#    selected_events = selected_events[:max_snippets]
#    
#    tot_frames = predictions.shape[0]
#    windows2read = []
#    for ev in selected_events:
#        l = max(0, ev - snippet_offset)
#        if l + snippet_size > tot_frames:
#            l = tot_frames - snippet_size
#        windows2read.append((l, l+snippet_size))
#    
        
        
        
        