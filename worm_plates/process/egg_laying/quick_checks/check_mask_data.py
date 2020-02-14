#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:39:07 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt

from scipy.spatial.distance import pdist, squareform
from extract_fixed_rois import link_eggs_full_frames
from plot_events import read_rois

if __name__ == '__main__':
    
    bn = 'CX11315_worms10_food1-10_Set6_Pos5_Ch1_18082017_123129'
    
    mask_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos/'
    
    
    #mask_file = list(mask_dir.rglob(f'*{bn}.hdf5'))[0]
    mask_file = '/Users/avelinojaver/workspace/WormData/screenings/CeNDR/MaskedVideos/CeNDR_Set3_180817/CX11315_worms10_food1-10_Set6_Pos5_Ch1_18082017_123129.hdf5'
    
    eggs_dir = Path.home() / 'workspace/WormData/egg_laying_test/worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64/'
    
    
    eggs_file = eggs_dir / f'{bn}_eggs_mask.csv'
    eggs_mask_frames = pd.read_csv(eggs_file)
    
    eggs_file = eggs_dir / f'{bn}_eggs_full.csv'
    eggs_full_frames = pd.read_csv(eggs_file)
    eggs_trajectories = link_eggs_full_frames(eggs_full_frames, max_dist = 50)
    eggs2check = [x[0] for x in eggs_trajectories if x[0]['frame_number'] > 0]
    #%%
    eggs_by_frame = {frame_number:frame_data.to_records(index=False) for frame_number, frame_data in eggs_mask_frames.groupby('frame_number')}
    
    #%%
    full_frame_interval = 5000
    
    max_dist = 5
    
    all_trajs = []
    for seed in eggs2check:
        t0, t1 =  (seed['frame_number'] -1)*full_frame_interval, seed['frame_number']*full_frame_interval
        
        
        traj = [(t1, seed['x'], seed['y'])] 
        for frame in range(t1-1, t0, -1):
            if not frame in eggs_by_frame:
                continue
            
            eggs_in_frame = eggs_by_frame[frame]
            dx = traj[-1][1] - eggs_in_frame['x']
            dy = traj[-1][2] - eggs_in_frame['y']
            
            r = np.sqrt(dx*dx + dy*dy)
            good = r < max_dist
            if np.any(good):
                ind = np.argmin(r)
                
                row = eggs_in_frame[ind]
                traj.append((frame, row['x'], row['y']))
        all_trajs.append(traj)
        
    egg_layings = [x[-1] for x in all_trajs]
    
    egg_layings = pd.DataFrame(egg_layings, columns = ['frames', 'x', 'y'])
    
    events_rois = read_rois(egg_layings, mask_file)
    
    #%%
    for vid, snippet in events_rois.items():
        fig, axs = plt.subplots(1, len(snippet), figsize = (15, 3), sharex = True, sharey = True)
        
        for ax, (frame, roi, egg) in zip(axs, snippet):
            ax.imshow(roi, cmap = 'gray')
            ax.axis('off')
            ax.set_title(frame)
    #%%
#    with tables.File(mask_file) as fid:
#        img = fid.get_node('/full_data')[-1]
#        
#    plt.figure(figsize=(15,15)) 
#    plt.imshow(img,cmap='gray')
#    
#    
#    plt.plot(1209.0, 1290.0, 'r.')
    
#    for frame_number, frame_data in eggs_full_frames.groupby('frame_number'):
#        img = imgs[frame_number]
#        
#        plt.figure(figsize = (10, 10))
#        plt.imshow(img, cmap='gray')
#        plt.plot(frame_data['x'], frame_data['y'], '.r')
#    
    #%%
    #plt.figure(figsize = (10, 10))
    #plt.imshow(img, cmap='gray')
    #plt.plot(eggs_mask_frames['x'], eggs_mask_frames['y'], '.')
    #plt.plot(eggs_full_frames['x'], eggs_full_frames['y'], '.r')
    
    
#    eggs_l = link_eggs_full_frames(eggs_mask_frames, max_dist = 5)
#    plt.figure(figsize = (10, 10))
#    plt.imshow(img, cmap='gray')
#    for tt in eggs_l:
#        if len(tt) > 1:
#            tt = np.array(tt)
#            x, y = tt['x'] , tt['y']
#            plt.plot(x,y,'-')
#    
    
    
    
    
    
    