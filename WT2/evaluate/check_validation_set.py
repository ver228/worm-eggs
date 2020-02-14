#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:29:45 2020

@author: avelinojaver
"""
import sys
from pathlib import Path
_src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(_src_dir))
from process.process_files_all import load_model
from train.flow import SnippetsFullFlow

from pathlib import Path
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    
    #model_base = 'WT+mixed-setups_unet-v4+R_20200131_171115_adam-_lr5e-05_wd0.0_batch32'
    #model_base = 'WT+mixed-setups_unet-v4_20200131_171116_adam-_lr0.0001_wd0.0_batch32'
    model_base = 'WT+mixed-setups_unet-v4+R_BCEp2_20200205_120934_adam-_lr5e-05_wd0.0_batch32'
    
    model_path = model_path_root / model_base / 'model_best.pth.tar'
    
    model, model_args = load_model(model_path)
    
    
    data_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data'
    #test_dir = data_dir / 'mixed-setups' / 'test'
    test_dir = data_dir / 'mixed-setups' / 'validation'
    assert test_dir.exists()
    
    val_flow = SnippetsFullFlow(test_dir)
    #%%
    th_true = 0.85
    
    metrics = np.zeros(4)
    pbar = tqdm.tqdm(val_flow)
    for image, target in pbar:
        
        #if not target.any():
        #    continue
        
        #images = images.to(device)
        #target = target.to(device)
        with torch.no_grad():
            X = torch.from_numpy(image[None])
            prediction = model(X)
        
        #loss = criterion(prediction, target)
        #test_avg_loss += loss.item()
        
        
        
        
        prediction = prediction[0].squeeze(-1)
        prediction = torch.sigmoid(prediction) # TODO once you figure out what model to use...
        prediction = prediction.detach().cpu().numpy()
        
        target_true = (target >= th_true)
        prediction_true = (prediction >= th_true)
        
        is_correct = target_true == prediction_true
        
        true_pos = is_correct[target_true].sum()
        true_neg = is_correct[~target_true].sum()
        
        total_pos = target_true.sum()
        total_neg = len(target) - total_pos
        
        metrics += (true_pos, total_pos, true_neg, total_neg)
        
        if true_neg != total_neg:
            
            fig, axs = plt.subplots(1, len(image), sharex = True, sharey = True)
            for ax, img, t, p in zip(axs, image, target_true, prediction_true):
                ax.imshow(img)
                ax.set_title(f'T{t}:P{p}')
                
            
        #%%
                
            
            
            
            
        
        
        