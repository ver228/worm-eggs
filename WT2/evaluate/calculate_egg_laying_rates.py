#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:55:01 2019

@author: avelinojaver
"""
import sqlalchemy

from tqdm import tqdm
import pandas as pd
import numpy as np

from pathlib import Path

if __name__ == '__main__':
    clearly_bad = ['egl-42 (n995) on food L_2011_05_25__10_03_52__1', 'npr-8 (tm1553) on food R_2010_01_26__15_39_59__9']
    
    predictions_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4/'
    src_file = predictions_dir / f'egg_laying_events_th0.8.csv'
    save_file = predictions_dir / f'egg_laying_rates.csv'
    
    cnx= sqlalchemy.create_engine('mysql+pymysql://root@localhost/single_worm_db')
    
    
    df_egg_laying = pd.read_csv(src_file)
    df_egg_laying = df_egg_laying[~df_egg_laying['basename'].isin(clearly_bad)]
    
    
    #df_info = pd.read_sql('SELECT * FROM experiments_full')
    df_info = pd.read_sql('SELECT base_name, strain, strain_description, results_dir FROM experiments_full WHERE developmental_stage = "adult" AND arena = "35mm petri dish NGM agar low peptone" AND sex = "hermaphrodite" AND food = "OP50" AND habituation = "30m wait"', cnx)
    
    #%%
    
    df = pd.merge(df_egg_laying, df_info, how='inner',
         left_on='basename', right_on='base_name')
    
    #%%
    def _get_laying_rate(dat):
        tot_time = dat['total_time'].sum()/60
        tot_events = dat['n_events'].sum()
        
        _rate = tot_events/tot_time    
        return _rate
    
    
    results = []
    for strain, strain_data in tqdm(df.groupby('strain')):
        if len(strain_data) < 10:
            continue
        
        rate = _get_laying_rate(strain_data)
        tot_time = strain_data['total_time'].sum()/60
        tot_events = strain_data['n_events'].sum()
        
        
        inds = strain_data.index
        
        
        samples = []
        for _ in range(25):
            inds_sampled = np.random.choice(inds, inds.size)
            dat_sampled = strain_data.loc[inds_sampled]
            rate_sample = _get_laying_rate(dat_sampled)
            samples.append(rate_sample)
        rate_std = np.std(samples)
        
        strain_description = strain_data.iloc[0]['strain_description']
        
        row = (strain, strain_description, rate, rate_std, len(strain_data), tot_time, tot_events)
        
        results.append(row)
        
    results_df = pd.DataFrame(results, columns = ['strain', 'strain_description', 'egg_laying_rate[min]', 'egg_laying_SD', 'tot_videos', 'tot_time[min]', 'tot_events'])
    results_df = results_df.sort_values(by = 'egg_laying_rate[min]')
    
    #%%
    results_df.to_csv(save_file, index = False, float_format='%.4f')
    
    
        
    #%%
        