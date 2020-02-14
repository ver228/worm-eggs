#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:31:20 2019

@author: avelinojaver
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tables
from tqdm import tqdm
from scipy.signal import find_peaks
if __name__ == '__main__':
    
    snippet_size = 11
    snippet_offset = 3
    min_separation = 5
    max_snippets = 20
    
    #valid_size = [(120,-120), (200,-200)]
    #valid_size = [(150,-150), (230,-230)]
    valid_size = [(135,-135), (215,-215)]
    
    #src_dir = Path.home() / 'workspace/WormData/egg_laying/predictions/WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4/train/'
    #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/data/v2+hard-neg/train'
    #save_prefix = 'HARDNEG'
    
    #src_dir = Path.home() / 'workspace/WormData/egg_laying/predictions/WT+v2+hard-neg_unet-v1_20190823_153141_adam-_lr0.0001_wd0.0_batch4/train/'
    #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/data/v2+hard-neg-2/train'
    #save_prefix = 'HARDNEGV2'

#    src_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4/'
#    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v3+hard-neg/'
#    save_prefix = 'HARDNEG'
#    thresh2check = 0.8
    
    src_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions_bkp/WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4/'
    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4/'
    #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4_240pix/'
    save_prefix = 'HARDNEG'
    thresh2check = 0.8
    
    # src_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions_bkp/WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8/'
    # root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4/'
    # #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4_240pix/'
    # save_prefix = 'HARDNEGV2'
    # thresh2check = 0.6
    
    
    annotations_df = pd.read_csv('single_user_labels/egg_events_corrected.csv')
    
    
    annotations = {bn : (dat.iloc[0]['set_type'], dat['frame_number'].values)
                   for bn, dat in annotations_df.groupby('base_name') 
                   }
    
    
    
    root_save_dir.mkdir(exist_ok = True, parents=True)
    
    src_files = [x for x in Path(src_dir).glob('*.p')]
    for src_file in tqdm(src_files):
        bn = src_file.stem[:-5]
        if not bn in annotations:
            continue
        set_type, true_eggs = annotations[bn]
         
        
        with open(src_file, 'rb') as fid:
            results = pickle.load(fid)
        
        predictions = results['predicted_egg_flags']
        pred_on, props = find_peaks(predictions, height = thresh2check)
        
        true_flags = np.zeros_like(predictions)
        true_flags[true_eggs] = 1
        
        valid_with_offset = np.concatenate((true_eggs - 1, true_eggs, true_eggs+1))
        wrong_events = set(pred_on) - set(valid_with_offset)
        
        pred_with_offset = np.concatenate((pred_on - 1, pred_on, pred_on+1))
        valid_events = set(true_eggs) & set(pred_with_offset)
        missing_events = set(true_eggs) - valid_events
        
        
        valid_events = list(valid_events)
        wrong_events = list(wrong_events)
        missing_events = list(missing_events)
        
        if not wrong_events:
            continue
        
        wrong_events = sorted(wrong_events)
        selected_events = [wrong_events[0]]
        for w in wrong_events[1:]:
            if w - selected_events[-1] >= min_separation:
                selected_events.append(w)
        selected_events = sorted(selected_events, key = lambda x : predictions[x], reverse = True)
        selected_events = selected_events[:max_snippets]
        
        tot_frames = predictions.shape[0]
        windows2read = []
        for ev in selected_events:
            l = max(0, ev - snippet_offset)
            if l + snippet_size > tot_frames:
                l = tot_frames - snippet_size
            windows2read.append((l, l+snippet_size))
        
        video_file = results['video_file']
        video_file = Path.home() / results['video_file'].replace('/users/rittscher/avelino/', '')
        
        with tables.File(video_file) as fid:
            masks = fid.get_node('/mask')
            
            dat = []
            for l,r in windows2read:
                (xl, xr), (yl, yr) = valid_size
                snippet = masks[l:r, xl:xr, yl:yr]
                egg_flag = true_flags[l:r]
                
                dat.append((snippet, egg_flag))
        
        snippets, egg_flags = zip(*dat)
        
        snippets = np.array(snippets, dtype = np.float32)
        egg_flags = np.array(egg_flags, dtype = np.float32)
        
        filters = tables.Filters(complevel = 5, complib = 'blosc:lz4')
        bn = Path(video_file).stem
        save_name = root_save_dir / set_type /  f'{save_prefix}_{bn}.hdf5'
        save_name.parent.mkdir(exist_ok = True)
        
        with tables.File(save_name, 'w') as fid:
            fid.create_carray('/', 'snippets', obj = snippets, filters = filters)
            fid.create_carray('/', 'egg_flags', obj = egg_flags, filters = filters)
        