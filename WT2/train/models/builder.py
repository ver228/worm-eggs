#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:24:35 2019

@author: avelinojaver
"""

from .model_unet import EggLayingDetectorUnetV1, EggLayingDetectorUnetV2, EggLayingDetectorUnetV3, EggLayingDetectorUnetV4
from .model_simple import EggLayingDetectorV1, EggLayingDetectorV2, EggLayingDetectorV3

from functools import partial

def get_model(model_name, n_in = 1, n_out = 2, **argkws):
    if model_name.endswith('+R'):
        model_name = model_name[:-2]
    
    if model_name == 'simple-v1':
        model_obj = EggLayingDetectorV1
    elif model_name == 'simple-v2':
        model_obj = EggLayingDetectorV2
    elif model_name == 'simple-v3':
        model_obj = EggLayingDetectorV3
    elif model_name == 'unet-v1':
        model_obj = EggLayingDetectorUnetV1
    elif model_name == 'unet-v2':
        model_obj = EggLayingDetectorUnetV2
    elif model_name == 'unet-v3':
        model_obj = EggLayingDetectorUnetV3
    elif model_name == 'unet-v3-bn':
        model_obj = partial(EggLayingDetectorUnetV3, batch_norm = True)
    elif model_name == 'unet-v4':
        model_obj = EggLayingDetectorUnetV4
    else:
        raise ValueError(f'Not Implemented `{model_name}`.')
    
    return model_obj(n_in, n_out, **argkws)