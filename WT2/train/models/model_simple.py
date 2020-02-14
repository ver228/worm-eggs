#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:26:21 2019

@author: avelinojaver
"""
from cell_localization.models.unet.unet_base import ConvBlock, DownSimple, init_weights
from torch import nn
from functools import partial

#%%
class ModelBase(nn.Module):
    def __init__(self, 
                 initial_block,
                 down_blocks,
                 pad_mode = 'constant'
                 ):
        super().__init__()
        
        
        self.initial_block = initial_block
        self.down_blocks = nn.ModuleList(down_blocks)
        
        #used only if the image size does not match the corresponding unet levels
        self.pad_mode = pad_mode
        self.n_levels = len(down_blocks)
    
       
   
    def forward(self, xin):
        
        
        x = self.initial_block(xin)
        
        x_downs = [x]
        for down in self.down_blocks:
            x = down(x)
            x_downs.append(x)
        
        
        return x
        
def base_constructor(n_inputs,
                     DownBlock = None,
                     InitialBlock = None,
                     initial_filter_size = 48, 
                     levels = 4, 
                     conv_per_level = 2,
                     increase_factor = 2,
                     batchnorm = False,
                     
                     pad_mode = 'constant'
                     ):
    
    down_filters = []
    nf = initial_filter_size
    for _ in range(levels):
        nf_next = int(nf*increase_factor)
        filters = [nf] + [nf_next]*conv_per_level
        nf = nf_next
        down_filters.append(filters)
    
    if DownBlock is None:
        DownBlock = DownSimple

    if InitialBlock is None:
        InitialBlock = partial(ConvBlock, kernel_size = 7)

    
    initial_block = InitialBlock(n_inputs, initial_filter_size, batchnorm = batchnorm)
    down_blocks = [DownBlock(x, batchnorm = batchnorm) for x in down_filters]
    
    n_outputs = down_filters[-1][-1]    
    model = ModelBase(initial_block, down_blocks,  pad_mode = pad_mode)
    
    return model, n_outputs


class EggLayingDetectorV3(nn.Module):
    def __init__(self,  n_inputs,  n_classes, embedding_size = 128, init_type = 'xavier', **arkgws):
        super().__init__()
        
        self.mapping_network, n_filters_map = base_constructor(n_inputs, **arkgws)
        
        self.pool = nn.Sequential(
                nn.Conv2d(n_filters_map, n_filters_map, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_filters_map, embedding_size, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.AdaptiveMaxPool2d(1)
                )
        
        self.conv2d_1d = nn.Sequential(
                nn.Conv3d(n_filters_map, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                
                nn.Conv3d(512, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                
                
                nn.Conv3d(256, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                
                nn.AdaptiveMaxPool3d((1, 1, None))
                )
        
        self.fc_clf = nn.Linear(64, n_classes)
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
    
    def forward(self, xin):
        n_batch, snippet_size, h, w = xin.shape
        
        #the model is expected to be in 2d conv. This because it was pretrained on fixed images
        xin = xin.view(-1, 1, h, w)
        x = self.mapping_network(xin)
        
        n_filts, h, w = x.shape[-3:]
        x = x.view(n_batch, snippet_size, n_filts, h, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.conv2d_1d(x)
        
        x = x.permute(0, 4, 1, 2, 3)
        n_batch, snippet_size, n_filts, h, w = x.shape
        x = x.view(n_batch, snippet_size, n_filts)
        x = self.fc_clf(x)
        
        return x

class EggLayingDetectorV2(nn.Module):
    def __init__(self,  n_inputs,  n_classes, embedding_size = 128, init_type = 'xavier', **arkgws):
        super().__init__()
        
        self.mapping_network, n_filters_map = base_constructor(n_inputs, **arkgws)
        
        self.pool = nn.Sequential(
                nn.Conv2d(n_filters_map, n_filters_map, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_filters_map, embedding_size, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.AdaptiveMaxPool2d(1)
                )
        
        self.clf = nn.Sequential(
                nn.Conv1d(embedding_size, 128, kernel_size = 5, padding = 2),
                nn.Tanh(),
                nn.Conv1d(128, 64, kernel_size = 3, padding = 1),
                nn.Tanh(),
                nn.Conv1d(64, n_classes, kernel_size = 3, padding = 1),
                )
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)

    def forward(self, xin):
        n_batch, snippet_size, h, w = xin.shape
        
        #the model is expected to be in 2d conv. This because it was pretrained on fixed images
        xin = xin.view(-1, 1, h, w)
        x = self.mapping_network(xin)
        
        x = self.pool(x)
        x = x.view(n_batch, snippet_size, -1)
        x = x.transpose(1,2)
        
        xout = self.clf(x)
        xout = xout.transpose(1,2)
        
        return xout


class EggLayingDetectorV1(nn.Module):
    def __init__(self,  n_inputs,  n_classes, embedding_size = 64, dropout_p = 0.1, init_type = 'xavier', **arkgws):
        super().__init__()
        
        self.mapping_network, n_filters_map = base_constructor(n_inputs, **arkgws)
        self.pool = nn.Sequential(
                ConvBlock(n_filters_map, embedding_size, batchnorm = True),
                nn.LeakyReLU(negative_slope=0.1),
                nn.AdaptiveAvgPool2d(1)
                )
        
        self.attn = nn.MultiheadAttention(embedding_size, num_heads = 2, dropout = dropout_p)
        
        self.clf = nn.Sequential(
                nn.Linear(embedding_size, 32),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(32, n_classes),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.1)
                )
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
        
    def forward(self, X):
        
        n_batch, snippet_size, h, w = X.shape
        xin = X.view(-1, 1, h, w)
        
        x = self.mapping_network(xin)
        x = self.pool(x)
        
        #flatten the singleton dimensions
        x = x.view(n_batch, snippet_size, -1)
        
        #the MultiheadAttention requires the dimensions to be (sequence, batch, embeddings) so we need to switch the first two dimensions
        x = x.transpose(0, 1)
        attn_output, attn_output_weights = self.attn(x, x, x) #self attention
        attn_output = attn_output.transpose(0, 1)
    
        xout = self.clf(attn_output)
        return xout


#%%
if __name__ == '__main__':
    import torch
    
    X = torch.rand((1,5,320,240))
    
    model = EggLayingDetectorV3(1, 2)
    xout = model(X)