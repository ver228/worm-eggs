#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:04:54 2019

@author: avelinojaver
"""
from pathlib import Path
from tqdm import tqdm
import tables
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def read_rois(df, 
              mask_file, 
              frames_str = 'frame_number',
              roi_size = 48, 
              offsets = [-25, -2, 0, 2, 25], 
              img_shape = (2048, 2048)
              ):
    #%%
    half_roi = roi_size // 2
    
    df['id'] = df.index
    
    data2read = []
    for tt in offsets:
        df_offset = df.copy() 
        df_offset[frames_str] += tt
        data2read.append(df_offset)
    
    data2read = pd.concat(data2read)
    data2read.loc[data2read[frames_str]<0, frames_str] = 0

    events_rois = defaultdict(list)
    
    with tables.File(mask_file) as fid:
        masks = fid.get_node('/mask')
        for frame_number, data_frame in data2read.groupby(frames_str):
            img = masks[int(frame_number)]
            
            for irow, row in data_frame.iterrows():
                xl = max(0, int(row['x'] - half_roi))
                xl = min(xl, img_shape[0]-1)
                yl = max(0, int(row['y'] - half_roi))
                yl = min(yl, img_shape[1]-1)
                
                roi = img[yl:yl+roi_size, xl:xl+roi_size].copy()
                
                event_id = int(row['id'])
                events_rois[event_id].append((frame_number, roi, (row['x'] - xl, row['y'] - yl)))
                
    #%%
    return events_rois

if __name__ == '__main__':
    
    root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos'
    mask_files = [x for x in root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
    files_dict = {x.stem : x for x in mask_files}
    #%%
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    
    
    events_dir = Path.home() / 'workspace/WormData/egg_laying_test/' / bn
    save_dir = Path.home() / 'workspace/WormData/egg_laying_plots/' / bn
    save_dir.mkdir(parents = True, exist_ok = True)
    
    ext = '_eggs_events_filtered.csv'
    events_files = [x for x in events_dir.rglob('*' + ext)]
    #events_files = [x for x in events_dir.rglob('CB4856_worms10_food1-10_Set2_Pos5_Ch1_02062017_121709_eggs_events.csv')]
    
    
    
    img_shape = (2048, 2048)
    
    max_dist = 10
    
    all_events = []
    for events_file in tqdm(events_files):
        events_df = pd.read_csv(events_file)
        events_df = events_df.drop_duplicates()
        events_df = events_df[events_df['closest_dist'] < max_dist]
        events_df = events_df[(events_df['closest_segment'] >= 26) & (events_df['closest_segment'] <= 31)]
        
        all_events.append(events_df)
        bn = events_file.name[:-len(ext)]
        mask_file = files_dict[bn]
        
        events_rois = read_rois(events_df, mask_file)
        for event_id, snippet in events_rois.items():
            fig, axs = plt.subplots(1, len(snippet), figsize = (15, 3), sharex = True, sharey = True)
            for ax, (frame, roi, egg) in zip(axs, snippet):
                ax.imshow(roi, cmap = 'gray')
                ax.axis('off')
                ax.set_title(int(frame))
            
            
            row = events_df.loc[event_id]
            
            t = int(row['frame_number'])
            w = int(row['worm_index'])
            save_name = save_dir / f'{bn}_{event_id}_{w}_{t}.png'
            
            fig.savefig(save_name)
            plt.close()
                
            
            
        
        
        