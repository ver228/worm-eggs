#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:04:25 2019

@author: avelinojaver
"""

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt
from scipy.signal import find_peaks

def timings2list(timings):
    if timings != timings:
        timings_l = []
    else:
        timings_l = [float(x) for x in timings.split(';')]
    return timings_l


def _old():
    max_dist = 3
    user_weights = {'aexbrown' : (1., 1085)}
    
    _log = ['aexbrown']
    
    for _ in tqdm(range(10), desc = 'iteration'):
        #estimate control events
        ctr_users = [k for k, (w, n) in user_weights.items() if n >= 3 and w >= 0.9]
        if _log[-1] == ctr_users:
            break
        
        _log.append(ctr_users)
        
        #%%
        ctr_rows = df_full['user_name'].isin(ctr_users)
        df_valid = df_full[ctr_rows]
        
        #%%
        keys2check = {}
        
        df_valid_g = df_valid.groupby(['file_name', 'start_point'])
        for k, dat in tqdm(df_valid_g, total = len(df_valid_g), desc = 'Getting Timings...'):
            timings = []
            for _, row in dat.iterrows():
                t = tuple(timings2list(row['timings']))
                timings.append((user_weights[row['user_name']][1], t))
                
            keys2check[k] = max(timings)[-1]
        #%%
        
        users_scores = defaultdict(list)
        for k, ref_timings in tqdm(keys2check.items(), desc = 'Scoring ...'):
            inds = df_grouped[k]
            
            for _, row in df_full.loc[inds].iterrows():
                v_timings = timings2list(row['timings'])
            
                if not ref_timings:
                    score = -len(v_timings) if v_timings else 1
                else:
                    
                    score = 0
                    for ref in ref_timings:
                        if v_timings:
                            dist = [(abs(x-ref), x) for x in v_timings]
                            dist_min = min(dist)
                            if dist_min[0] < max_dist:
                                t = dist_min[1]
                                v_timings.remove(t)
                                score += 1
                        else:
                            score -= 1
                            
                    score -= len(v_timings)
                    
                    users_scores[row['user_name']].append(score)
        
        user_weights = {}
        for k, v in sorted(users_scores.items()):
            v = np.array(v)
            avg_weight = np.mean(v>0)
            user_weights[k] = (avg_weight, v.size)
        
        #%%
        user_weights_df = [(k,*v) for k,v in user_weights.items()]
        user_weights_df = pd.DataFrame(user_weights_df, columns = ['user', 'weight', 'n_scored'])
        
        save_name = Path(fname).parent / 'user_weights.csv'
        user_weights_df.to_csv(save_name)
        
        break

if __name__ == '__main__':
    #%%
    _debug = True
    fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/2015-10-25_worms_classifications.csv'
    save_name = '/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/consensus_events.csv'
    
    
    min_videos = 3
    
    
    df_full = pd.read_csv(fname)
    #%%
    df_full = df_full[['file_name', 'start_point', 'user_name', 'timings', 'video_length']]
    df_full = df_full.drop_duplicates()
    
    df_full = df_full[~df_full['file_name'].isnull()]
    
    counts_per_user = df_full['user_name'].value_counts()
    #%%
    valid_users = counts_per_user.index[counts_per_user > min_videos]
    df_full = df_full[df_full['user_name'].isin(valid_users)]
    
    df_grouped = df_full.groupby(['file_name', 'start_point']).groups
    
    #%%
    empty_segments = []
    identified_events = {}
    
    fps = 1/30
    peak_threshold = 0.5
    
    
    pool_window_seconds = 0.5 # I am planning to pool all the signals within a window...
    pool_window = int(round(pool_window_seconds/fps)) 

    keys2check = [(k, v) for k,v in df_grouped.items() if len(v) > 3]
    
    rows2save = []
    for key, inds in tqdm(keys2check):
        segment_size_seconds = df_full.loc[inds[0], 'video_length'] 
        
        # i am expecting windows of around 30 seconds, the exact size is not very important as long as it is larger than the buffer
        segment_size_frames = int(round(segment_size_seconds/fps)) + 1
        
        flags = np.zeros((len(inds), segment_size_frames))
        for irow, (_, row) in enumerate(df_full.loc[inds].iterrows()):
            t = np.array(timings2list(row['timings']))
            t = t[ t<= segment_size_seconds] #there seem to some events labeled at times larger than the `video_length`. I am removing those
            
            if t.size > 0:
                t = np.unique(t)
                
                frames = np.round(t/fps).astype(np.int)
                
                flags[irow, frames] = 1
        
        
        avg_counts = flags.sum(axis=1)
        
        med = np.median(avg_counts)
        mad = np.median(np.abs(avg_counts - med))
        
        th = (med - 3*mad), (med + 3*mad)
        good = (avg_counts >= th[0]) & (avg_counts <= th[1])
        flags = flags[good] 
        
        flags_pooled = flags.sum(axis=0)
        flags_pooled = np.convolve(flags_pooled, np.ones(pool_window))
        flags_pooled /= flags.shape[0]
        
        
        basename, start_point = key
        
        if flags_pooled.sum() == 0:
            peaks_string = 'NULL'
        else:
            peaks_inds, props = find_peaks(flags_pooled, height = peak_threshold, distance = pool_window/2)
            peaks_times = start_point + peaks_inds*fps
            peaks_string = ';'.join([str(x) for x in peaks_times])
            
            if _debug:
                #%%
                plt.figure()
                plt.plot(flags.mean(axis=0))
                plt.plot(flags_pooled)
                plt.plot(peaks_inds, props['peak_heights'], 'xr')        
                plt.title(key)
                
                #%%
                if len(plt.get_fignums()) > 100:
                    break
            
            
        
        basename = basename[:-8]
        row2save = (basename, start_point, segment_size_seconds, peaks_string)
        rows2save.append(row2save)
    
    if not _debug:
    
        df = pd.DataFrame(rows2save, columns = ['basename', 'start_point', 'segment_size', 'events'])
        df.to_csv(save_name, index = False)

            
        
            
            