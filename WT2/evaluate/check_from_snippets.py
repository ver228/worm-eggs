#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:57:11 2020

@author: avelinojaver
"""
import sys
from pathlib import Path
_src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(_src_dir))
from process.process_files_all import load_model
from train.flow import SnippetsFullFlow

from pathlib import Path
import pickle
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    import matplotlib.pylab as plt

    #model_base = 'WT+mixed-setups_unet-v4+R_20200131_171115_adam-_lr5e-05_wd0.0_batch32'
    model_base = 'WT+mixed-setups_unet-v4+R_BCEp2_20200205_120934_adam-_lr5e-05_wd0.0_batch32'
    
    src_root_dir = Path.home() / 'workspace/WormData/egg_laying/plates/annotations2check/syngenta/'
    dir_snippets = src_root_dir / model_base / 'snippets'

    model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    model_path = model_path_root / model_base / 'model_best.pth.tar'
    model, model_args = load_model(model_path)
    
    #%%
    fnames = list(dir_snippets.glob('*.pickle'))
    #%%
    fnames = [x for x in fnames if '31130' in x.name]
    for fname in tqdm.tqdm(fnames):
        
        with open(fname, 'rb') as fid:
            dat = pickle.load(fid)
    
        t = 12
        
        #m = 40
        #X = dat[None, m:-m, t:-t, t:-t]
        X = dat[None, :,  t:-t, t:-t]
        
        
        X = X.astype(np.float32)/255
        
        #X = np.concatenate((X, X), axis = 1)
        X = torch.from_numpy(X)
        
        with torch.no_grad():
            prediction = model(X)
            
        prediction = prediction[0].squeeze(-1)
        prediction = torch.sigmoid(prediction) # TODO once you figure out what model to use...
        prediction = prediction.detach().cpu().numpy()
        
        dd = X[0].detach().numpy()
        fig, axs = plt.subplots(1, dd.shape[0], figsize = (20, 5), sharex = True, sharey = True)
        for ax, roi, p in zip(axs, dd, prediction):
           ax.imshow(roi[t:-t, t:-t])
           ax.set_title(f'{p:.2f}')
           
        
        
        break
        
        
        
        
        
        