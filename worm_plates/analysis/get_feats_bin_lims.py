#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:35:15 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    events_dir = Path.home() / 'workspace/WormData/egg_laying_test/' / bn
    save_dir = events_dir / 'histograms_data'
    save_dir.mkdir(parents = True, exist_ok = True)
    
    events_f_files = [x for x in events_dir.rglob('*_timeseries.pkl')]
    
    q_bins = (0.005, 0.995)
    
    all_bin_ranges = []
    for ts_file in tqdm(events_f_files):
        df_ts = pd.read_pickle(ts_file)
        q = df_ts.quantile(q_bins)
        all_bin_ranges.append(q)
    
    q_bot = [q.loc[q_bins[0]] for q in all_bin_ranges]
    q_bot = pd.concat(q_bot, axis=1).min(axis=1)
    
    q_top = [q.loc[q_bins[1]] for q in all_bin_ranges]
    q_top = pd.concat(q_top, axis=1).max(axis=1)
    
    bin_ranges = pd.concat((q_bot, q_top), axis=1)
    bin_ranges.columns = ['bot', 'top']
    
    save_name = save_dir / 'bin_limits.csv'
    bin_ranges.to_csv(save_name)