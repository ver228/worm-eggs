#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:31:49 2019

@author: avelinojaver
"""

import pandas as pd
import tables
import numpy as np
from pathlib import Path
import tqdm

if __name__ == '__main__':
    missing_files = []
    
    bad_prefix = '/Volumes/behavgenom_archive$/single_worm/finished'
    root_data_dir = str(Path.home() / 'workspace/WormData/screenings/single_worm/finished/')
    
    
    #valid_size = [(150,-150), (230,-230)]
    #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4'#v1_0.5x'
    
    #valid_size = [(135,-135), (215,-215)]
    #root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4_210pix'
    
    valid_size = [(120,-120), (200,-200)]
    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4_240pix'
    
    
    (root_save_dir / 'test').mkdir(parents = True, exist_ok = True)
    (root_save_dir / 'train').mkdir(parents = True, exist_ok = True)
    
    df = pd.read_csv('single_user_labels/egg_events_corrected.csv')
    #%%
    snippet_size = 11
    half_size = snippet_size//2
    #resized_size = (320, 240)
    
   
    
    for bn, bn_data in tqdm.tqdm(df.groupby('base_name')):
        
        eggs = bn_data['frame_number'].values
    
        dname = bn_data['results_dir'].iloc[0]
        dname = dname.replace(bad_prefix, root_data_dir)
        fname = Path(dname) / (bn + '.hdf5')
        
        if not fname.exists():
            missing_files.append(fname)
            continue
        
        data = []
        with tables.File(fname) as fid:
            masks = fid.get_node('/mask')
            tot_frames = masks.shape[0]
            
            eggs = eggs[eggs < tot_frames]
            
            true_flags = np.zeros(tot_frames)
            true_flags[eggs] = 1
            
            def get_limits(ind):
                l, r = ind - half_size, ind + half_size + 1
                if l < 0:
                    offset = abs(l)
                    r += offset
                    l = 0
                
                if r > tot_frames:
                    offset = r - tot_frames
                    l -= offset
                    r = tot_frames
                    
                return l,r
            
            
            
            flags_negatives = np.ones(tot_frames, np.bool)
            for egg in eggs:
                l, r = max(egg - snippet_size, 0), min(egg + snippet_size + 1, tot_frames )
                flags_negatives[l:r] = False
                
                l, r = get_limits(egg)
                
                (xl, xr), (yl, yr) = valid_size
                positive_snippet = masks[l:r, xl:xr, yl:yr]
                #positive_snippet = np.array([cv2.resize(x, resized_size) for x in positive_snippet])
                
                
                positive_flags = true_flags[l:r]
                
                data.append((positive_snippet, positive_flags))
                
                
#                import matplotlib.pylab as plt
#                fig, axs = plt.subplots(1, snippet_size, figsize = (3*snippet_size, 5), sharex = True, sharey = True)
#                
#                cc = np.convolve(positive_flags, [0, 0.15, 1., 0.5, 0.25], 'same')
#                for ax, ss, flag in zip(axs, positive_snippet, cc):
#                    ax.imshow(ss)
#                    ax.set_title(flag)
#                    
#                plt.suptitle((egg, bn))
#                break
#                
            #%%
                #%%
                
            eggs_negatives, = np.where(flags_negatives)
            eggs_negatives = np.random.choice(eggs_negatives, 10)
            for egg in eggs_negatives:
                l, r = get_limits(egg)
                negative_snippet = masks[l:r]
                
                (xl, xr), (yl, yr) = valid_size
                negative_snippet = masks[l:r, xl:xr, yl:yr]
                negative_flags = true_flags[l:r]
                
                data.append((negative_snippet, negative_flags))
          
        snippets, egg_flags = map(lambda x : np.array(x, np.float32), zip(*data))
        
        set_type = bn_data['set_type'].iloc[0]
        save_name = root_save_dir / set_type / f'EGGLAYING_{bn}.hdf5'
        
        try:
            filters = tables.Filters(complevel = 5, complib = 'blosc:lz4')
            with tables.File(save_name, 'w') as fid:
                fid.create_carray('/', 'snippets', obj = snippets, filters = filters)
                fid.create_carray('/', 'egg_flags', obj = egg_flags, filters = filters)
        except:
            raise
            import pdb
            pdb.set_trace()
    