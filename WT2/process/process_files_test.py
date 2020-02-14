#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:54:11 2019

@author: avelinojaver
"""

from process_files_all import process_files_all
from tqdm import tqdm
import random

if __name__ == '__main__':
    bn2check = [
                 'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4',
                 'WT+v3+hard-neg_unet-v4+R_20191028_205224_adam-_lr0.0001_wd0.0_batch8',
                 'WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8',
                 'WT+v4_unet-v4_20191029_153801_adam-_lr0.0001_wd0.0_batch10_frame-shift+kernel-simple',
                 'WT+v4_unet-v4_20191029_181002_adam-_lr0.0001_wd0.0_batch10_frame-shift',
                 'WT+v4_unet-v4_20191030_143735_adam-_lr0.0001_wd0.0_batch10_intensity+kernel-simple',
                 'WT+v4_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v2+hard-neg-2_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v4_unet-v4_20191101_160320_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v4_unet-v4+hardaug_20191031_214855_adam-_lr0.0001_wd0.0_batch10',
                 
                 'WT+v4_unet-v4+R_20191102_175649_adam-_lr0.0001_wd0.0_batch10',
                 'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
                ]
    
    random.shuffle(bn2check)
    for bn in tqdm(bn2check):
        snippet_size = 160 if 'unet-v4' in bn else 100
        
        process_files_all(
            snippet_size = snippet_size,
            model_base = bn,
            is_only_test = True
            )