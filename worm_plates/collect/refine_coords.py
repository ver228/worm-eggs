#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:22:37 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
import tables
import tqdm
import cv2

import numpy as np
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
#%%
def correct_coords(img_, coords_, min_distance = 1, max_dist = 5):
    #%%
    peaks = peak_local_max(img_, min_distance = min_distance)
    peaks = peaks[:, ::-1]
    
    #remove `peaks` that is not close by to any `coord` by at most `max_dist`
    D = cdist(coords_, peaks)
    
    
    #peaks with an intensity smaller than the coords intensities will be spurious
    peaks_ints = img_[peaks[:, 1], peaks[:, 0]]
    cc = coords_.astype(np.int)
    coords_int = img_[cc[:, 1], cc[:, 0]]
    
    good = (D <= max_dist).any(axis=0)
    good &= peaks_ints >= coords_int.min()
    
    D = D[:, good]
    valid_peaks = peaks[good]
    
    
    
    
    #find the closest peaks
    closest_indexes = np.argmin(D, axis=1)
    
    #we will consider as an easy assigment if the closest peak is assigned to only one coord
    u_indexes = np.unique(closest_indexes)
    counts = np.bincount(closest_indexes)[u_indexes]
    easy_assigments = u_indexes[counts == 1]
    
    valid_pairs = [(ii, x) for ii, x in enumerate(closest_indexes) if x in easy_assigments]
    
    if len(valid_pairs) > 0:
        easy_rows, easy_cols = map(np.array, zip(*valid_pairs))
        
        easy_cost = D[easy_rows, easy_cols]
        good = easy_cost<max_dist
        easy_rows = easy_rows[good]
        easy_cols = easy_cols[good]
        
        assert (D[easy_rows, easy_cols] <= max_dist).all()
        
        #now hard assigments are if a peak is assigned to more than one peak
        ambigous_rows = np.ones(D.shape[0], np.bool)
        ambigous_rows[easy_rows] = False
        ambigous_rows, = np.where(ambigous_rows)
        
        
        ambigous_cols = np.ones(D.shape[1], np.bool)
        ambigous_cols[easy_cols] = False
        ambigous_cols, = np.where(ambigous_cols)
    else:
        ambigous_rows = np.arange(D.shape[0])
        ambigous_cols = np.arange(D.shape[1])
        easy_rows = np.array([], dtype=np.int)
        easy_cols = np.array([], dtype=np.int)
    
    D_r = D[ambigous_rows][:, ambigous_cols]
    good = (D_r <= max_dist).any(axis=0)
    D_r = D_r[:, good]
    ambigous_cols = ambigous_cols[good]
    
    #for this one we use the hungarian algorithm for the assigment. This assigment is to slow over the whole matrix
    ri, ci = linear_sum_assignment(D_r)
    
    hard_rows, hard_cols = ambigous_rows[ri], ambigous_cols[ci]
    
    assert (D_r[ri, ci] == D[hard_rows, hard_cols]).all()
    
    hard_cost = D[hard_rows, hard_cols]
    good = hard_cost<max_dist
    hard_rows = hard_rows[good]
    hard_cols = hard_cols[good]
    
    
    #let's combine both and assign the corresponding peak
    rows = np.concatenate((easy_rows, hard_rows))
    cols = np.concatenate((easy_cols, hard_cols))
    
    new_coords = coords_.copy()
    new_coords[rows] = valid_peaks[cols] #coords that do not satisfy the close peak condition will not be changed
    
    return new_coords


#%%
if __name__ == '__main__':
    _debug = False
    
    min_distance = 2
    max_dist = 5  
    
    r = max_dist*2+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
    
    src_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam/'
    dst_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam_refined/'
    
    
    src_files = [x for x in src_root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
    
    
    for src_file in tqdm.tqdm(src_files):
        
        with pd.HDFStore(src_file, 'r') as fid:
            df = fid['/coords']
            img = fid.get_node('/img')[:]
        #%%
        #create a mask using the known coordinates
        valid_mask = np.zeros_like(img)
        cols = df['cx'].astype(np.int)
        rows = df['cy'].astype(np.int)
        valid_mask[rows, cols] = 1
        
        
        valid_mask = cv2.dilate(valid_mask, kernel) > 0
        
        #then I will use the inverted maxima to to create local maxima corresponding to the refined eggs peaks
        img_peaks = ~img
        img_peaks -= img_peaks[valid_mask].min()
        img_peaks[~valid_mask] = 0
        #img_peaks = cv2.blur(img_peaks, (1,1))
        #%%
        #finaly use the correct coords function to assing each labelled coords to a local maxima
        cc = df[['cx','cy']].values
        new_coords = correct_coords(img_peaks, cc, min_distance, max_dist)
        
        
        coords = pd.DataFrame({'type_id':1, 'cx':new_coords[:,0], 'cy':new_coords[:,1]})
        coords = coords.to_records(index=False)
        
        dst_file = str(src_file).replace(str(src_root_dir), str(dst_root_dir))
        dst_file = Path(dst_file)
        dst_file.parent.mkdir(exist_ok=True, parents=True)
        with tables.File(str(dst_file), 'w') as fid:
            fid.create_carray('/', 'img', obj = img)
            fid.create_table('/', 'coords', obj = coords)
        #%%
        
        if _debug:
            #%%
            import matplotlib.pylab as plt
        
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            
            axs[0].imshow(img, cmap = 'gray')
            axs[1].imshow(img_peaks, cmap = 'gray')
            
            for ax in axs:
                ax.plot(df['cx'], df['cy'], '.r')
                ax.plot(coords['cx'], coords['cy'], '.g')
                
            plt.show()
            #%%
            break
        

        