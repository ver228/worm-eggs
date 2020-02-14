#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:15:38 2019

@author: avelinojaver
"""

from pathlib import Path 

from cell_localization.utils import get_device
from cell_localization.models import get_model

import tqdm
import tables
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

def load_model(model_path, **argkws):
    model_path = Path(model_path)
    
    bn = model_path.parent.name
    parts = bn.split('_')
    model_type = parts[1]
    loss_type = parts[2]
    
    
    state = torch.load(model_path, map_location = 'cpu')
    
    model = get_model(model_type, 1, 1, loss_type, **argkws)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, state['epoch']


def collate_simple(batch):
    return tuple(map(list, zip(*batch)))

class DirTierpsyFlow(Dataset):
    def __init__(self, root_dir, field_name = '/full_data'):
        root_dir = Path(root_dir)
        
        mask_files = root_dir.rglob('*.hdf5')
        mask_files = [x for x in mask_files if not x.name.startswith('.')]
        
        self.root_dir = root_dir
        self.mask_files = mask_files
        
    
    def __getitem__(self, ind):
        mask_file = self.mask_files[ind]
        with tables.File(mask_file, 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
        
        imgs = imgs[:, None]
        imgs = imgs.astype(np.float32)/255
        
        frames = np.arange(imgs.shape[0])
        
        return imgs, mask_file, frames
    
    def __len__(self):
        return len(self.mask_files)

def main(cuda_id = 0, 
         screen_type = 'Drug_Screening'
         ):
    
    
    #where the masked files are located
    root_dir = Path.home() / 'workspace/WormData/screenings' / screen_type/ 'MaskedVideos/'
    
    #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
    #nms_threshold_rel = 0.2
    
    #bn = 'worm-eggs-adam+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190803_225943_adam_lr0.000128_wd0.0_batch64'
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    nms_threshold_rel = 0.25
    
    model_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/'/ bn.partition('+F')[0] /  bn / 'model_best.pth.tar'
    
    
    #where the predictions are going to be stored
    save_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / screen_type / bn 
    
    
    model_args = dict(nms_threshold_abs = 0.,
                            nms_threshold_rel = nms_threshold_rel,
                            pad_mode = 'reflect'
                            )
    
    device = get_device(cuda_id)
    model, epoch = load_model(model_path, **model_args)
    model = model.to(device)
    
    
    gen = DirTierpsyFlow(root_dir)
    loader = DataLoader(gen, batch_size = 1, num_workers = 4, collate_fn = collate_simple)
    
    save_dir = Path(save_dir)
    
    for batch in tqdm.tqdm(loader):
        for (imgs, mask_file, frames) in zip(*batch):
            
            preds_l = []
            for frame_number, xin in zip(frames, imgs):
                
                with torch.no_grad():
                    xin = torch.from_numpy(xin[None])
                    xin = xin.to(device)
                    predictions = model(xin)
                
                    predictions = predictions[0]
                    res = [predictions[x].detach().cpu().numpy() for x in ['coordinates', 'scores_abs', 'scores_rel']]
                
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
    
    