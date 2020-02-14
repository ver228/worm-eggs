#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:26:21 2019

@author: avelinojaver
"""

from cell_localization.models.unet import init_weights, get_mapping_network
from cell_localization.models import model_types
from torch import nn
import torch.nn.functional as F
#%%

class ReducedUnet(nn.Module):
    def __init__(self, **argkws):
        super().__init__()
        self.net = get_mapping_network(**argkws)
        self.n_output_filters = self.net.up_blocks[-1].n_filters[-1]
        delattr(self.net, 'output_block')
        
        
        
    def forward(self, x_input):
        net = self.net
        
        pad_, pad_inv_ = net.calculate_padding(x_input.shape[2:], (net.n_levels + 1))
        x_input = F.pad(x_input, pad_, mode = net.pad_mode)
        
        x = net.initial_block(x_input)
        
        x_downs = []
        for down in net.down_blocks:
            x_downs.append(x)
            x = down(x)
        
        feats = [x]
        for x_down, up in zip(x_downs[::-1], net.up_blocks):
            x = up(x, x_down)
            feats.append(x)
        
        return feats
        
        

class EggLayingDetectorUnetV1(nn.Module):
    def __init__(self,  n_inputs,  n_classes, init_type  = None):
        super().__init__()
        
        
        net_args = model_types['unet-simple']
        self.mapping_network = ReducedUnet(**net_args,
                                          n_inputs = n_inputs,
                                          n_ouputs = n_inputs,
                                          return_feat_maps = True)
        
        n_filters_map = self.mapping_network.n_output_filters
        self.conv2d_1d = nn.Sequential(
                nn.Conv3d(n_filters_map, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                
                nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                
                nn.AdaptiveMaxPool3d((1, 1, None))
                )
        
        self.fc_clf = nn.Linear(32, n_classes)
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
        
    def forward(self, xin):
        n_batch, snippet_size, h, w = xin.shape
        xin = xin.view(-1, 1, h, w)
        
        
        feats = self.mapping_network(xin)
        x = feats[-1]
        
        n_filts, h, w = x.shape[-3:]
        x = x.view(n_batch, snippet_size, n_filts, h, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.conv2d_1d(x)
        
        x = x.permute(0, 4, 1, 2, 3)
        n_batch, snippet_size, n_filts, h, w = x.shape
        x = x.view(n_batch, snippet_size, n_filts)
        x = self.fc_clf(x)
        
        return x


class EggLayingDetectorUnetV2(nn.Module):
    def __init__(self,  n_inputs,  n_classes, init_type  = None):
        super().__init__()
        
        
        
        net_args = model_types['unet-simple']
        self.mapping_network = ReducedUnet(**net_args,
                                          n_inputs = n_inputs,
                                          n_ouputs = n_inputs,
                                          return_feat_maps = True)
        
        n_filters_map = self.mapping_network.n_output_filters
        
        
        self.pool = nn.Sequential(
                nn.Conv2d(n_filters_map, 64, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.AdaptiveMaxPool2d(1)
                )
        
        self.clf = nn.Sequential(
                nn.Conv1d(64, 32, kernel_size = 5, padding = 2),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(32, n_classes, kernel_size = 5, padding = 2),
                )
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
        
    def forward(self, xin):
        n_batch, snippet_size, h, w = xin.shape
        xin = xin.view(-1, 1, h, w)
        
        
        feats = self.mapping_network(xin)
        x = feats[-1]
        
        x = self.pool(x)
        x = x.view(n_batch, snippet_size, -1)
        
        x = x.transpose(1,2)
        xout = self.clf(x)
        xout = xout.transpose(1,2)
        
        return xout

        
        return x


def get_conv2dplus1d(in_planes, out_planes, midplanes, batch_norm = False):

    #https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
    bias = not batch_norm
    layers = [
        nn.Conv3d(in_planes, 
                 midplanes, 
                 kernel_size=(1, 3, 3),
                 padding=(0, 1, 1),
                 bias = bias
                 )
    ]
    if batch_norm:
        layers += [nn.BatchNorm3d(midplanes)]


    layers += [
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, 
                      out_planes, 
                      kernel_size=(3, 1, 1),
                      padding=(1, 0, 0),
                      bias=bias)
            ]
    return layers



class EggLayingDetectorUnetV3(nn.Module):
    def __init__(self,  n_inputs,  n_classes, init_type  = None, batch_norm = False):
        super().__init__()
        
        if batch_norm:
            net_args = model_types['unet-simple-bn']
        else:
            net_args = model_types['unet-simple']

        self.mapping_network = ReducedUnet(**net_args,
                                          n_inputs = n_inputs,
                                          n_ouputs = n_inputs,
                                          return_feat_maps = True)
        




        n_filters_map = self.mapping_network.n_output_filters


        layers = get_conv2dplus1d(n_filters_map, 64, 64, batch_norm)  
        if batch_norm:
            layers += [nn.BatchNorm3d(64)] 
        layers += [nn.ReLU(inplace=True)] + get_conv2dplus1d(64, 32, 64, batch_norm) + [nn.AdaptiveMaxPool3d((1, 1, None))]


        self.conv2d_1d = nn.Sequential(*layers)
        
        self.fc_clf = nn.Linear(32, n_classes)
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
        
    def forward(self, xin):
        n_batch, snippet_size, h, w = xin.shape
        xin = xin.view(-1, 1, h, w)
        
        
        feats = self.mapping_network(xin)
        x = feats[-1]
        
        n_filts, h, w = x.shape[-3:]
        x = x.view(n_batch, snippet_size, n_filts, h, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.conv2d_1d(x)
        
        x = x.permute(0, 4, 1, 2, 3)
        n_batch, snippet_size, n_filts, h, w = x.shape
        x = x.view(n_batch, snippet_size, n_filts)
        x = self.fc_clf(x)
        
        return x

class EggLayingDetectorUnetV4(nn.Module):
    def __init__(self,  n_inputs,  n_classes, init_type  = None):
        super().__init__()
        
        net_args = model_types['unet-flat-48']

        self.mapping_network = ReducedUnet(**net_args,
                                          n_inputs = n_inputs,
                                          n_ouputs = n_inputs,
                                          return_feat_maps = True)
        




        n_filters_map = self.mapping_network.n_output_filters


        layers = get_conv2dplus1d(n_filters_map, 64, 64)  
        layers += [nn.ReLU(inplace=True)] + get_conv2dplus1d(64, 32, 64) + [nn.AdaptiveMaxPool3d((1, 1, None))]


        self.conv2d_1d = nn.Sequential(*layers)
        
        self.fc_clf = nn.Linear(32, n_classes)
        
        #initialize model
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
        
    def forward(self, xin):
        
        n_batch, snippet_size, h, w = xin.shape
        xin = xin.view(-1, 1, h, w)
        
        
        feats = self.mapping_network(xin)
        x = feats[-1]
        
        n_filts, h, w = x.shape[-3:]
        x = x.view(n_batch, snippet_size, n_filts, h, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.conv2d_1d(x)
        
        x = x.permute(0, 4, 1, 2, 3)
        n_batch, snippet_size, n_filts, h, w = x.shape
        x = x.view(n_batch, snippet_size, n_filts)
        x = self.fc_clf(x)
        
        return x

if __name__ == '__main__':
    import torch
    
    X = torch.rand((1, 15, 90, 90))
    #%%
    # model = EggLayingDetectorUnetV3(1, 1, batch_norm = False)
    
    # model = EggLayingDetectorUnetV3(1, 1, batch_norm = True)
    
    model = EggLayingDetectorUnetV4(1, 1)
    xout = model(X)
    
    
        
                    
                    