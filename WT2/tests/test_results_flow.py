#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:11:58 2019

@author: avelinojaver
"""
import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))

from train.models import get_model
from train.flow import SnippetsFullFlow


from tqdm import tqdm
from cell_localization.utils import  get_device
import torch
import tables

import pickle
import matplotlib.pylab as plt

if __name__ == '__main__':
    cuda_id = 0
    device = get_device(cuda_id)
    
    root_dir = Path.home() / 'workspace/WormData/egg_laying/data/v2/test'
    gen = SnippetsFullFlow(root_dir)
    #%%
    
    model_path = Path.home() / 'workspace/WormData/egg_laying/results/WT+v2_unet-v1+pretrained_20190819_170413_adam-_lr1e-05_wd0.0_batch4/model_best.pth.tar'
    state = torch.load(model_path, map_location = 'cpu')
    state_dict = {}
    for k,v in state['state_dict'].items():
        if 'output_block.' in k:
            continue
        if k.startswith('mapping_network.'):
            k = k[:16] + 'net.' + k[16:]
        state_dict[k] = v
        
    model = get_model('unet-v1', n_in = 1, n_out = 1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    #%%
    model.eval()
    
    batch_size = 96
    offset = 3
    
    results = []
    with torch.no_grad():
        for snippets, is_egg_laying in tqdm(gen):
             pass
            #%%
            
            
#            #%%
#            fig, axs = plt.subplots(1, len(snippets), figsize = (25, 5), sharex = True, sharey = True)
#            for ax, ss, flag in zip(axs, snippets, is_egg_laying):
#                ax.imshow(ss)
#                ax.set_title(flag)
#                masks = fid.get_node('/mask')
#            for images, target in pbar:
#                
#                imgs = masks[t:t+batch_size, xl:xr, yl:yr]
#                
#                X = torch.from_numpy(imgs[:, None])
#                X = X.float()/255
#                X = X.to(device)
#                
#                
#                prediction = model(X)
#                prediction = prediction[0].squeeze(-1)
#                prediction = torch.sigmoid(prediction)
#                prediction = prediction.detach().cpu().numpy()
#                
#                results.append((t, prediction))
#                
#    with open(save_name, 'wb') as fid:
#        pickle.dump(results, fid)
    