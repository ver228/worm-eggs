#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:26:21 2019

@author: avelinojaver
"""

if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    import torch

    from flow import SnippetsFullFlow
    from models import EggLayingDetectorUnet
    
    bn = 'WT2-egg-laying+pretrained_20190813_001621_adam-_lr1e-05_wd0.0_batch2'#'WT2-egg-laying+pretrained_20190812_224025_adam-_lr1e-05_wd0.0_batch2'
    model_path =  Path().home() / 'workspace/WormData/egg_laying/results/' / bn / 'model_best.pth.tar'
    
    
    print(bn)
    #%%
    
    model = EggLayingDetectorUnet(1, 2)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
   #%%
   
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/egg_laying/data/v1_0.5x/test'
    gen = SnippetsFullFlow(root_dir)
    #%%
    batch = []
    for ii in range(4):
        snippet, is_egg_laying  = gen[ii]
        batch.append((snippet, is_egg_laying))
    #%%
    snippet, is_egg_laying =  zip(*batch)
    X = torch.from_numpy(np.stack(snippet))
    y = torch.from_numpy(np.stack(is_egg_laying))
    
    #%%
    xout = model(X)
    
    vv, mm = xout.max(dim=-1)
    print(y)
    print(mm)
    print(vv)