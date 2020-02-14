#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:50:32 2017

@author: ajaver
"""


import random
import pandas as pd
import pymysql
import os
import shutil
#%%
def read_egg_events(fname = 'manually_verified.xlsx'):
    egg_lists = pd.read_excel(fname)
    all_eggs = []
    for ii, row in egg_lists.iterrows():
        row = row.dropna()
        row = row.values
        
        base_name =  row[0].replace('.hdf5', '')
        egg_frames = list(map(int, row[1:]))
        
        if len(egg_frames) > 0:
            all_eggs += zip([base_name]*len(egg_frames), egg_frames)
        
    egg_events = pd.DataFrame(all_eggs, columns=['base_name', 'frame_number'])
    return egg_events
#%%

dst_dir = '/data/ajaver/egg_laying/training_set'
if __name__ == '__main__':
    conn = pymysql.connect(user='ajaver', host='localhost', db = 'single_worm_db')
    cur = conn.cursor()
    
    root_file = ''
    egg_events = read_egg_events(fname = 'manually_verified.xlsx')
    #%%
    u_basenames = egg_events['base_name'].unique()
    
    random.seed(a=777)
    random.shuffle(u_basenames)
    part_ind = round(len(u_basenames)*0.2)
    
    test_bn = [(x, 'test') for x in u_basenames[:part_ind]]
    train_bn = [(x, 'train') for x in u_basenames[part_ind:]]
    
    bn2set = {f:d for f,d in test_bn + train_bn}
    egg_events['set_type'] = egg_events['base_name'].map(bn2set)
    #%% get the results directory
    f_str = ' ,'.join(['"{}"'.format(x) for x in u_basenames])
    sql = 'SELECT base_name, results_dir FROM experiments WHERE base_name IN ({});'.format(f_str)
    cur.execute(sql)
    results = cur.fetchall()
    
    bn2dir = {f:d for f,d in results}
    egg_events['results_dir'] = egg_events['base_name'].map(bn2dir)
    #%%
    bad_file = []
    feat_file_prev = ''
    for irow, row in egg_events.iterrows():
        print(irow)
        
        bn = os.path.join(row['results_dir'], row['base_name'])
        feat_file = bn + '_skeletons.hdf5'
        mask_file = bn + '.hdf5'
    
        
        if feat_file_prev != feat_file:
            
            try:
                feat_loc = os.path.join(dst_dir, row['base_name'] + '_skeletons.hdf5')
                with pd.HDFStore(feat_loc, 'r') as fid:
                    trajectories_data = fid['/trajectories_data']
                if not (trajectories_data.index == trajectories_data['frame_number']).all():
                    bad_file.append(feat_file)
            
            except:
                if not os.path.exists(feat_file):
                    bad_file.append(feat_file)
                    continue
                
                shutil.copy(feat_file, dst_dir)
                shutil.copy(mask_file, dst_dir)
                
        feat_file_prev = feat_file
    
    #%%
    bad = [os.path.basename(x).replace('_skeletons.hdf5', '') for x in bad_file]
    
    egg_events = egg_events[~egg_events['base_name'].isin(bad)]
    
    
    #%%
    egg_events.to_csv('egg_events.csv', index=False)
    
    
    
    