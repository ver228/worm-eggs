#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:45 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd

dname = Path(__file__).resolve().parents[1]
ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events_231019.tsv'

if __name__ == '__main__':
    df_true = []
    with open(ANNOTATIONS_FILE) as fid:
        for row in fid.read().split('\n'):
            if row:
                dd = row.split('\t')
                bn = dd[0]
                for event in dd[1:]:
                    df_true.append((bn, int(event)))
    df_true = pd.DataFrame(df_true, columns = ['base_name', 'frame_number'])

    features_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
    ts_files = [x for x in features_dir.rglob('*.hdf5') if not (x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
    ts_files = {x.stem:x for x in ts_files}
    
    #%%
    home_str = str(Path.home())
    
    fnames2save = [] 
    for bn in df_true['base_name'].unique():
        fname = str(ts_files[bn])
        fname = fname.replace(home_str, '$HOME')
        
        fnames2save.append(fname)
        
    str2save = '\n'.join(fnames2save)
    
    save_name = ANNOTATIONS_FILE.parent / 'egg_events_231019_files.txt'
    with open(save_name, 'w') as fid:
        fid.write(str2save)