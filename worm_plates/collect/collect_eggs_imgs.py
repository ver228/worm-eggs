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

#%%
if __name__ == '__main__':
    random.seed(777)
    
    #root_dir = Path.home() / 'workspace/WormData/screenings/Drug_Screening/MaskedVideos/'
    root_dir = '/Users/avelinojaver/Desktop/Adam_egg_data_for_Avelino'
    root_dir = Path(root_dir)
    
    save_dir = Path.home() / 'workspace/localization/data/worm_eggs/'
    
    fnames = [x for x in root_dir.rglob('*.csv') if not x.name.startswith('.')]
    
    #annotations = pd.DataFrame(annotations, columns = ['type', 'type_id', 'radius', 'cx', 'cy'])
    
    val_frac = 0.05
    test_frac = 0.03
    
    test_nn = math.ceil(len(fnames) * test_frac)
    val_nn = math.ceil(len(fnames) * test_frac)
    train_nn = len(fnames) - test_nn - val_nn
    
    set_types = ['test']*test_nn + ['train']*train_nn + ['val']*val_nn
    random.shuffle(set_types)
    
    fnames = [x for x in fnames if not x.name.startswith('.')]
    for fname, set_type in zip(tqdm.tqdm(fnames), set_types):
        df = pd.read_csv(str(fname))
        if len(df) == 0:
            continue
        
        bn = fname.name.split('_eggs')[0]
        #for frame_number, dat in df.groupby('frame_number'):
        frame_number = df['frame_number'].min()
        dat = df[df['frame_number'] == frame_number]
        
        img_name = fname.parent / f'{bn}_frame-{frame_number}.png'
        img = cv2.imread(str(img_name), -1)
        
        coords = pd.DataFrame({'type_id' : 1, 
                      'cx' : dat['x'], 
                      'cy' : dat['y']
                      })
        coords = coords.to_records(index=False)
        
        
        ss = save_dir / set_type
    
        prefix = str(fname.parent).replace(str(root_dir), '')[1:]
        save_name = ss / prefix / f'EGGS-frame-{frame_number}_{fname.stem}.hdf5'
        save_name.parent.mkdir(exist_ok=True, parents=True)
        with tables.File(str(save_name), 'w') as fid:
            fid.create_carray('/', 'img', obj = img)
            fid.create_table('/', 'coords', obj = coords)
        
            

        