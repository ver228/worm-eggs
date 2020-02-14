#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:15:38 2019

@author: avelinojaver
"""

from pathlib import Path 
from extract_eggs import load_model, get_device

from tqdm import tqdm
import tables
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class VideoTierpsyFlow(Dataset):
    def __init__(self, fname, field_name = '/mask'):
        self.fid = tables.File(fname, 'r')
        self.imgs = self.fid.get_node(field_name)
    
    def __getitem__(self, frame):
        img = self.imgs[frame]
        img = img[None].astype(np.float32)/255
        
        return frame, img
    
    def __len__(self):
        return self.imgs.shape[0]

def main(cuda_id = 0, 
         screen_type = 'Drug_Screening'
         ):
    
    
    #where the masked files are located
    root_dir = Path.home() / 'workspace/WormData/screenings' / screen_type/ 'MaskedVideos/'
    
    #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
    #nms_threshold_rel = 0.2
    
    bn = 'worm-eggs-adam+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190803_225943_adam_lr0.000128_wd0.0_batch64'
    nms_threshold_rel = 0.25
    
    model_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam/'/ bn / 'model_best.pth.tar'
    
    
    #where the predictions are going to be stored
    save_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / screen_type / bn 
    save_dir = Path(save_dir)
        
    
    model_args = dict(nms_threshold_abs = 0.,
                            nms_threshold_rel = nms_threshold_rel,
                            pad_mode = 'reflect'
                            )
    
    device = get_device(cuda_id)
    model, epoch = load_model(model_path, **model_args)
    model = model.to(device)
    
    mask_files = root_dir.rglob('*.hdf5')
    mask_files = [x for x in mask_files if not x.name.startswith('.')]
    
    
    for mask_file in tqdm(mask_files):
        gen = VideoTierpsyFlow(mask_file)
        loader = DataLoader(gen, batch_size = 1, num_workers = 1)
        
        preds_l = []
        for frames, imgs in tqdm(loader):
        
            with torch.no_grad():
                imgs = imgs.to(device)
                predictions = model(imgs)
            
            for frame_number, pred in zip(frames, predictions):
                res = [pred[x].detach().cpu().numpy() for x in ['coordinates', 'scores_abs', 'scores_rel']]
            
                res = [x[:, None] if x.ndim == 1 else x for x in res]
                res = np.concatenate(res, axis=1)
                preds_l += [(frame_number, *cc) for cc in zip(*res.T)]
                
        preds_df = pd.DataFrame(preds_l, columns = ['frame_number', 'x', 'y', 'score_abs', 'score_rel'])
        
        save_name = Path(str(mask_file).replace(str(root_dir), str(save_dir)))
        save_name = save_name.parent / (save_name.stem + '_eggs-preds.csv')
        save_name.parent.mkdir(exist_ok=True, parents=True)
        
        preds_df.to_csv(save_name, index = False)
        
    

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    
    