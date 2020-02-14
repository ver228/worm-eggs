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
import math
import random
import cv2
import numpy as np

#I copy this from tierpsy tracker, I do not want to create a new dependency for a single function
def getROIMask(
        image,
        min_area,
        max_area,
        thresh_block_size,
        thresh_C,
        dilation_size,
        keep_border_data,
        is_light_background):
    '''
    Calculate a binary mask to mark areas where it is possible to find worms.
    Objects with less than min_area or more than max_area pixels are rejected.
        > min_area -- minimum blob area to be considered in the mask
        > max_area -- max blob area to be considered in the mask
        > thresh_C -- threshold used by openCV adaptiveThreshold
        > thresh_block_size -- block size used by openCV adaptiveThreshold
        > dilation_size -- size of the structure element to dilate the mask
        > keep_border_data -- (bool) if false it will reject any blob that touches the image border

    '''


    # Objects that touch the limit of the image are removed. I use -2 because
    # openCV findCountours remove the border pixels
    IM_LIMX = image.shape[0] - 2
    IM_LIMY = image.shape[1] - 2

    #this value must be at least 3 in order to work with the blocks
    thresh_block_size = max(3, thresh_block_size)
    if thresh_block_size % 2 == 0:
        thresh_block_size += 1  # this value must be odd

    #let's add a median filter, this will smooth the image, and eliminate small variations in intensity
    # now done with opencv instead of scipy
    image = cv2.medianBlur(image, 5)

    # adaptative threshold is the best way to find possible worms. The
    # parameters are set manually, they seem to work fine if there is no
    # condensation in the sample
    if not is_light_background:  # invert the threshold (change thresh_C->-thresh_C and cv2.THRESH_BINARY_INV->cv2.THRESH_BINARY) if we are dealing with a fluorescence image
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            thresh_block_size,
            -thresh_C)
    else:
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            thresh_block_size,
            thresh_C)

    # find the contour of the connected objects (much faster than labeled
    # images)
    _, contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find good contours: between max_area and min_area, and do not touch the
    # image border
    goodIndex = []
    for ii, contour in enumerate(contours):
        if not keep_border_data:
            # eliminate blobs that touch a border
            keep = not np.any(contour == 1) and \
                not np.any(contour[:, :, 0] ==  IM_LIMY)\
                and not np.any(contour[:, :, 1] == IM_LIMX)
        else:
            keep = True

        if keep:
            area = cv2.contourArea(contour)
            if (area >= min_area) and (area <= max_area):
                goodIndex.append(ii)

    # typically there are more bad contours therefore it is cheaper to draw
    # only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 1, cv2.FILLED)

    # drawContours left an extra line if the blob touches the border. It is
    # necessary to remove it
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[:, -1] = 0

    # dilate the elements to increase the ROI, in case we are missing
    # something important
    struct_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    mask = cv2.dilate(mask, struct_element, iterations=3)

    return mask
#%%
    

    
if __name__ == '__main__':
    
    DFLT_ARGS = dict(
            min_area = 25, 
            max_area = int(1e8), 
            thresh_block_size = 61, 
            thresh_C = 15, 
            dilation_size = 5, 
            keep_border_data = False, 
            is_light_background = True
            )
    
    
    root_dir = Path.home() / 'workspace/WormData/Adam_eggs/annotated/'
    #root_dir = Path('/Users/avelinojaver/Desktop/Adam_egg_data_for_Avelino')
    
    save_dir = Path.home() / 'workspace/localization/data/worm_eggs_with_masks/'
    
    fnames = [x for x in root_dir.rglob('*.csv') if not x.name.startswith('.')]
    
    
    ref_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam/'
    ref_partition = {x.stem[:-5].partition('_')[-1]:str(x.parent).replace(str(ref_dir), '').split('/')[1] for x in ref_dir.rglob('*.hdf5')}
    
    #val_frac = 0.05
    #test_frac = 0.03
    #test_nn = math.ceil(len(fnames) * test_frac)
    #val_nn = math.ceil(len(fnames) * test_frac)
    #train_nn = len(fnames) - test_nn - val_nn
    #set_types = ['test']*test_nn + ['train']*train_nn + ['val']*val_nn
    #random.shuffle(set_types)
    
    fnames = [x for x in fnames if not x.name.startswith('.')]
    for fname in tqdm.tqdm(fnames):
        df = pd.read_csv(str(fname))
        if len(df) == 0:
            continue
        
        
        
        bn = fname.name.split('_eggs')[0]
        
        if not bn in ref_partition:
            print(bn)
        else:
            set_type = ref_partition[bn]
        
        #%%
        
        #for frame_number, dat in df.groupby('frame_number'):
        frame_number = df['frame_number'].min()
        dat = df[df['frame_number'] == frame_number]
        
        img_name = fname.parent / f'{bn}_frame-{frame_number}.png'
        img = cv2.imread(str(img_name), -1)
        if img is None:
            raise ValueError(img_name)
        
        coords = pd.DataFrame({'type_id' : 1, 
                      'cx' : dat['x'], 
                      'cy' : dat['y']
                      })
        coords = coords.to_records(index=False)
        
        #%%
        mask = getROIMask(img, **DFLT_ARGS)
        img_masked = img.copy()
        img_masked[mask == 0] = 0
        
        #let's reduce the size to only include the area with valid pixels
        ind_x, = np.where(mask.sum(axis=0))
        ind_y, = np.where(mask.sum(axis=1))
        mask = mask[ind_y[0]:ind_y[-1]+1, ind_x[0]:ind_x[-1]+1]
        img_masked = img_masked[ind_y[0]:ind_y[-1]+1, ind_x[0]:ind_x[-1]+1]
        
        mask_coords = []
        for cx,cy in zip(dat['x'] - ind_x[0], dat['y']- ind_y[0]):
            x, y = int(cx), int(cy)
            
            xl, xr = max(0, x-5), x + 5
            yl, yr = max(0, y-5), y + 5
            
            roi = mask[yl:yr, xl:xr]
            
            p = (roi >0).mean()
            if p > 0.8:
                mask_coords.append((1, cx, cy))
            
        mask_coords = pd.DataFrame(mask_coords, columns=['type_id', 'cx', 'cy'])
        mask_coords = mask_coords.to_records(index=False)
        #%%
        
        ss = save_dir / set_type
        prefix = str(fname.parent).replace(str(root_dir), '')[1:]
        save_name = ss / prefix / f'EGGS-frame-{frame_number}_{fname.stem}.hdf5'
        
        save_name.parent.mkdir(exist_ok=True, parents=True)
        with tables.File(str(save_name), 'w') as fid:
            fid.create_carray('/', 'img', obj = img)
            fid.create_table('/', 'coords', obj = coords)
        
        if mask_coords.size > 0:
            save_name = ss / prefix / f'EGGS-frame-{frame_number}_{fname.stem}_mask.hdf5'
            save_name.parent.mkdir(exist_ok=True, parents=True)
            with tables.File(str(save_name), 'w') as fid:
                fid.create_carray('/', 'img', obj = img_masked)
                fid.create_table('/', 'coords', obj = mask_coords)
        