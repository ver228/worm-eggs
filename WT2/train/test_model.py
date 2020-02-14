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
    from model import EggLayingDetector
    
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    model_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam-masks/' / bn / 'model_best.pth.tar'
    
    nms_threshold_abs = 0.0
    nms_threshold_rel = 0.25
    loss_type = 'maxlikelihood'
    model_type = 'clf+unet-simple'
    n_ch_in = 1
    n_classes = 2
    
    
    
    #%%
    model = EggLayingDetector(n_ch_in, n_classes)
    model.eval()
   #%%
   
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/egg_laying/data/v1_0.5x/test'
    gen = SnippetsFullFlow(root_dir)
    #%%
    batch = []
    for _ in range(4):
        snippet, is_egg_laying  = gen[0]
        batch.append((snippet, is_egg_laying))
    #%%
    snippet, is_egg_laying =  zip(*batch)
    X = torch.from_numpy(np.stack(snippet))
    y = torch.from_numpy(np.stack(is_egg_laying))
    
    #%%
    xout = model(X)
    