#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:07:53 2019

@author: avelinojaver
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

if __name__ == '__main__':
    #bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    
    #root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/Results'
    #events_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/CeNDR' / bn
    
    root_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam'
    events_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/syngenta' / bn
    
    #%%
    ext = '_featuresN.hdf5'
    feats_files = [x for x in root_dir.rglob('*' + ext) if not x.name.startswith('.')]
    files_dict = {x.name[:-len(ext)] : x for x in feats_files}
    
    
    events_files = [x for x in events_dir.rglob('*_eggs_events.csv') if  not x.name.startswith('.')]
    
    #%%
    max_dist_midbody = 10
    midbody_ranges = [26, 31]
    
    img_shape = (2048, 2048)
    for events_file in tqdm(events_files):
        events_df = pd.read_csv(events_file)
        k = events_file.stem[:-12]
        feats_file = files_dict[k]
        
        events_filtered_file = events_file.parent / (k + '_eggs_events_filtered.csv')
        
        filtered_events = []
        with pd.HDFStore(feats_file, 'r') as fid:
            microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
            blob_features = fid['/blob_features']
            trajectories_data = fid['/trajectories_data']
            skeletons = fid.get_node('/coordinates/skeletons')
            for irow, egg in events_df.iterrows():
                frame = int(egg['frames'])
                good = trajectories_data['frame_number'] == frame
                blobs_in_frame = blob_features.loc[good, ['coord_x', 'coord_y', 'box_length']]
                blobs_in_frame /= microns_per_pixel 
                
                half_roi =  blobs_in_frame['box_length'] / 2
                valid = egg['x'] > blobs_in_frame['coord_x'] - half_roi
                valid &= egg['x'] < blobs_in_frame['coord_x'] + half_roi
                valid &= egg['y'] > blobs_in_frame['coord_y'] - half_roi
                valid &= egg['y'] < blobs_in_frame['coord_y'] + half_roi
            
                valid_inds = blobs_in_frame[valid].index
                
                skels2check = []
                for centre_ind in valid_inds:
                    w_ind = trajectories_data.loc[centre_ind, 'worm_index_joined']
                    
                    l, r = max(0, centre_ind-10), centre_ind+10
                    valid_traj = trajectories_data.loc[l:r]
                    
                    good = (valid_traj['worm_index_joined'] == w_ind) & (valid_traj['skeleton_id'] >= 0) & (valid_traj['was_skeletonized']>0)
                    valid_traj = valid_traj[good]
                    
                    if valid_traj.size > 0:
                    
                        ind2check = (valid_traj['frame_number'] - frame).abs().idxmin()
                        skel_id = int(valid_traj.loc[ind2check, 'skeleton_id'])
                        skel = skeletons[skel_id]
                        
                        
                        if np.isnan(skel[0,0]):
                            #the previously selected skeleton is invalid, search among all the available skeletons...
                            
                            skels = skeletons[valid_traj['skeleton_id'].values, :]
                            good = ~np.isnan(skels[:, 0, 0])
                            valid_traj = valid_traj[good]
                            if valid_traj.size > 0:
                                ind2check = (valid_traj['frame_number'] - frame).abs().idxmin()
                                skel_id = int(valid_traj.loc[ind2check, 'skeleton_id'])
                                skel = skeletons[skel_id]
                            
                        
                        if not np.isnan(skel[0,0]):
                            skel = skel[midbody_ranges[0]:midbody_ranges[1]+1]/microns_per_pixel
                            
                            dx = (skel[:, 0] - egg['x'])
                            dy = (skel[:, 1] - egg['y'])
                            r = np.sqrt(dx*dx + dy*dy)
                            
                            ind_min = r.argmin()
                            rmin = r[ind_min]
                            
                            if rmin <= max_dist_midbody:
                                centre_offset = ind2check - centre_ind
                                skels2check.append((rmin, ind_min + midbody_ranges[0], centre_offset, w_ind))
                    
                if len(skels2check) > 0:
                    r_min, r_min_ind, c_offset, w_ind = min(skels2check)
                    filtered_events.append((w_ind, r_min, r_min_ind, c_offset, egg['frames'], egg['x'], egg['y']))
        
        if not filtered_events:
            filtered_events = np.zeros((0, 7))
        
        df = pd.DataFrame(filtered_events, columns = ['worm_index', 'closest_dist', 'closest_segment', 'centre_offset', 'frame_number', 'x', 'y'])
        df.to_csv(events_filtered_file, index = False)               

                