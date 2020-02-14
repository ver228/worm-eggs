#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:49:21 2019

@author: avelinojaver
"""

from pathlib import Path
from extract_fixed_rois import get_egg_laying_events, link_eggs_full_frames
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd

bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
src_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/syngenta' / bn

max_dist_btw_full_frames = 50
max_dist_btw_mask_frames = 5
full_frame_interval = 5000

src_files = [x for x in src_dir.rglob('*_eggs_full.csv') if not x.name.startswith('.')]
#%%
def _process_file(file_full):
    bn = file_full.name[:-14]
    file_mask = src_dir / (bn + '_eggs_mask.csv')
    assert file_mask.exists()

    eggs_full_frames = pd.read_csv(file_full)
    eggs_mask_frames = pd.read_csv(file_mask)
    
    
    eggs_trajectories = link_eggs_full_frames(eggs_full_frames, max_dist = max_dist_btw_full_frames)
    eggs2check = [x[0] for x in eggs_trajectories if x[0]['frame_number'] > 0]
    eggs_by_frame = {frame_number:frame_data.to_records(index=False) for frame_number, frame_data in eggs_mask_frames.groupby('frame_number')}
        
    egg_laying_events = get_egg_laying_events(eggs2check, eggs_by_frame, full_frame_interval, max_dist = max_dist_btw_mask_frames)
    
    
    save_name_events = save_name_events = src_dir / (bn + '_eggs_events.csv')
    egg_laying_events.to_csv(save_name_events, index = False)


with mp.Pool(8) as pool:
    for _ in tqdm(pool.imap(_process_file, src_files), total=len(src_files)):
        pass