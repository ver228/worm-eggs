#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:49:31 2019

@author: avelinojaver
"""
import tables
import matplotlib.pylab as plt
from pathlib import Path

src_dir = Path.home() / 'workspace/WormData/egg_laying/data/v2+hard-neg-2/train'
files2plot = list(src_dir.rglob('HARDNEGV2_*.hdf5'))[10:30]

for fname in files2plot[:10]:
    with tables.File(fname) as fid:
        snippets = fid.get_node('/snippets')[:]

#%%
    for snippet in snippets:
        
        fig, axs = plt.subplots(1, len(snippet), figsize = (20, 3))
        for ax, ss in zip(axs, snippet):
            ax.imshow(ss, cmap = 'gray')
            ax.axis('off')
        

