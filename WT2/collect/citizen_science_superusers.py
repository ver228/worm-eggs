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
    _debug = False
    fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/2015-10-25_worms_classifications.csv'
    save_name = '/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/consensus_events.csv'
    
    gold_user = 'aexbrown'
    
    
    df_full = pd.read_csv(fname)
    #%%
    df_full = df_full[['file_name', 'start_point', 'user_name', 'timings', 'video_length']]
    df_full = df_full.drop_duplicates()
    df_full = df_full[~df_full['file_name'].isnull()]
    #%%
    
    gold_flag = df_full['user_name'] == gold_user
    df_gold = df_full[gold_flag]
    df_remain = df_full[~gold_flag]
    
    
    gold_timings = {}
    for _, row in df_gold.iterrows():
        k = row['file_name'], row['start_point']
        gold_timings[k] = np.array(timings2list(row['timings']))
    
    #%%
    df_remain_g = df_remain.groupby(['file_name', 'start_point']).groups
    #%%
    min_videos = 10
    
    scores_per_user = {}
    for key, gold_t in tqdm(gold_timings.items()):
        inds = df_remain_g[key]
        
        #timings_per_user = {}
        dat = df_remain.loc[inds]
        for _, row in dat.iterrows():
            v_timings = timings2list(row['timings'])
            v_timings = np.array(v_timings)
            
            user_name = row['user_name']
            if not user_name in scores_per_user:
                scores_per_user[user_name] = [0, 0, []]
            
            #total scores per user
            scores_per_user[user_name][0] += 1
            
            #good negative
            if not len(gold_t):
                if not len(v_timings):
                    scores_per_user[user_name][1] += 1
            else:
                if len(v_timings):
                    min_dist = np.min(np.abs(v_timings[None, :] - gold_t[:, None]), axis=0)
                    scores_per_user[user_name][2].append(min_dist)
                
    
    scores_per_user = {k:v for k,v in scores_per_user.items() if v[0] >= min_videos}
    #%%
    for k, v in scores_per_user.items():
        if len(v[-1]) > 0:
            offsets = np.concatenate(v[-1])
            max_offset = offsets.max()
            med_offset = np.median(offsets)
            print(k, max_offset, med_offset)
    
    
    
    
    
    
    