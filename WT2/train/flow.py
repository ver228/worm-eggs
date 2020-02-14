#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:24:06 2019

@author: avelinojaver
"""
import tables
import os
from pathlib import Path
import numpy as np
import tqdm
from collections import defaultdict
import random
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, snippet, target):
        for t in self.transforms:
            snippet, target = t(snippet, target)
        return snippet, target

class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, snippet, target):
        if random.random() < self.prob:
            height, width = snippet.shape[:2]
            snippet = snippet[:, :, ::-1]
            
        return snippet, target

class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, snippet, target):
        if random.random() < self.prob:
            height, width = snippet.shape[-2:]
            snippet = snippet[:, ::-1, :]            
        return snippet, target


class RandomTemporalCrop(object):
    def __init__(self, snippet_size = 5):
        self.snippet_size = snippet_size

    def __call__(self, snippet, target):
        assert snippet.shape[0] > self.snippet_size
        assert not np.all(target)
        
        invalid_t, = np.where(target) #I do not want to start an snippet during an egg laying event, since the temporal information is important here
        for _ in range(50):
            l = random.randint(0, snippet.shape[0] - self.snippet_size -1)
            if not l in invalid_t:
                break
        else:
            raise ValueError('Cannot find a valid time to crop. Maybe there is a vector with only laying events')
        
        
        r = l + self.snippet_size
        
        snippet = snippet[l:r]
        target = target[l:r]
        
        assert snippet.shape[0] == self.snippet_size
        return snippet, target

class CenterTemporalCrop(object):
    def __init__(self, snippet_size):
        self.snippet_size = snippet_size

    def __call__(self, snippet, target):
        assert snippet.shape[0] >= self.snippet_size
        
        ind = snippet.shape[0]//2
        half = self.snippet_size//2
        
        l, r = ind-half, ind+half+1
        snippet = snippet[l:r]
        target = target[l:r]
        
        assert snippet.shape[0] == self.snippet_size
        return snippet, target


class RandomOffset(object):
    def __init__(self, max_offset = 10):
        self.max_offset = max_offset
        
    def __call__(self, snippet, target):
        x_offset = random.randint(-self.max_offset, self.max_offset)
        y_offset = random.randint(-self.max_offset, self.max_offset)
        
        snippet = np.roll(snippet, y_offset, axis = 1)
        snippet = np.roll(snippet, x_offset, axis = 1)
        
        return snippet, target


class RandomOffsetPerFrame(object):
    def __init__(self, max_offset = 25, prob = 0.5):
        self.max_offset = max_offset
        self.prob = prob
        
    def __call__(self, snippet, target):
        
        for ii in range(len(snippet)):
            if self.max_offset is not None and random.random() < self.prob:
            
                x_offset = random.randint(-self.max_offset, self.max_offset)
                y_offset = random.randint(-self.max_offset, self.max_offset)
                
                snippet[ii] = np.roll(snippet[ii], y_offset, axis = 1)
                snippet[ii] = np.roll(snippet[ii], x_offset, axis = 1)
        
        
        
        return snippet, target

class RandomZoom(object):
    def __init__(self, zoom_range = (0.95, 1.1)):
        self.zoom_range = zoom_range
        
    def __call__(self, snippet, target):
        
        
        z = random.uniform(*self.zoom_range)
        
        snippet_zoomed = np.array([cv2.resize(x, dsize=(0,0), fx=z, fy=z) for x in snippet])
        
        current_size = snippet_zoomed.shape[1:]
        target_size = snippet.shape[1:]
        
        delta_w = target_size[1] - current_size[1]
        delta_h = target_size[0] - current_size[0]
    
        top = delta_h//2
        bottom =  delta_h-top
        
        left = delta_w//2
        right = delta_w - left
        
        
        if z < 1:
            pad = [(0, 0), (bottom, top), (left, right)]
            fill_val = np.median(snippet_zoomed[0, 0, :])
            #snippet_zoomed = np.pad(snippet_zoomed, pad, 'edge')
            snippet_zoomed = np.pad(snippet_zoomed, pad, 'constant', constant_values = fill_val)  
        else:
            bottom, top = abs(bottom), snippet_zoomed.shape[1] + top
            left, right = abs(left), snippet_zoomed.shape[2] + right
            
            snippet_zoomed = snippet_zoomed[:, bottom:top, left:right]
        
        
        return snippet_zoomed, target

class NormalizeIntensity(object):
    def __init__(self, scale = (0, 255)):
        self.scale = scale
    
    def __call__(self, image, target):
        
        image = (image.astype(np.float32) - self.scale[0])/(self.scale[1] - self.scale[0])
        
        return image, target

class RandomIntensityOffset(object):
    def __init__(self, offset_range = (-0.2, 0.2)):
        self.offset_range = offset_range
    
    def __call__(self, image, target):
        if self.offset_range is not None and random.random() > 0.5:
            offset = random.uniform(*self.offset_range)
            image = image + offset
            
        return image, target


class RandomIntensityExpansion(object):
    def __init__(self, expansion_range = (0.7, 1.3)):
        self.expansion_range = expansion_range
        
    def __call__(self, image, target):
        if self.expansion_range is not None and random.random() > 0.5:
            factor = random.uniform(*self.expansion_range)
            image = image*factor
            
            
        return image, target    


class RandomErosion(object):
    def __init__(self, prob_global = 0.5, prob_per_frame = 0.5, kernel_size = 11):
        self.prob_global = prob_global
        self.prob_per_frame = prob_per_frame
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    
    def __call__(self, image, target):
        if random.random() < self.prob_global:
            masks = (image > 0).astype('uint8')
            for ii in range(len(image)):
                if random.random() < self.prob_per_frame:
                    masks[ii] = cv2.erode(masks[ii], self.kernel)
            image *= masks
            
        return image, target

class RandomFillBackground(object):
    def __init__(self, prob = 0.25, background_val = 0, background_percentile = 85):
        self.prob = prob
        self.background_val = background_val
        self.background_percentile = background_percentile
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            is_bgnd = image == self.background_val
            valid_pix = image[~is_bgnd]
            if valid_pix.size > 0:
                q = random.uniform(95, 100)
                val = np.percentile(valid_pix, q)
                
                image = image.copy()
                image[is_bgnd] = val
            
        return image, target
    
class RandomMotionBlur(object):
    def __init__(self, displacement_range = (5, 45), prob = 0.25):
        self.prob = prob
        self.displacement_range  = displacement_range

    def __call__(self, snippet, target):
        if self.displacement_range is not None and random.random()  < self.prob:
            ind = random.randint(0, snippet.shape[0] - 2)
            
            
            for ii in range(random.choice(( 2, ))): 
                ind2change = ind + ii
                
                
                ss = snippet[ind2change]
                kernel_size = random.randint(*self.displacement_range)
                angle = random.randint(-90, 90)
                
                kernel = np.zeros((kernel_size, kernel_size)) 
                kernel[kernel_size//2, :] = 1.
                
                
                rot_mat = cv2.getRotationMatrix2D( (kernel_size / 2 -0.5 , kernel_size / 2 -0.5 ) , angle, 1.0)
                kernel = cv2.warpAffine(kernel,  rot_mat, (kernel_size, kernel_size))
                
                kernel /= kernel.sum()
                blurred_image = cv2.filter2D(ss, -1, kernel)
                snippet[ind2change] = blurred_image
            
        return snippet, target


class ConvolveEggFlag(object):
    #kernel = [0, 0.15, 1., 0.6, 0.25]
    #kernel = [0, 0.15, 1., 0.3, 0.15]
    #kernel = [0, 0.25, 1., 0.45, 0.25]
    def __init__(self, kernel = [0, 0.15, 1., 0.6, 0.25]):
        self.kernel = np.array(kernel, np.float32)
    
    def __call__(self, image, target):
        target = np.convolve(target, self.kernel, 'same')
        target = np.clip(target, 0, 1)
        return image, target    
#%%
class SnippetsRandomFlow():
    def __init__(self, 
                 root_dir,
                 positive_prob = 0.5,
                 snippet_size = 7,
                 max_offset = 25,
                 max_offset_per_frame = None, #50,
                 motion_blur_range = None, #(5, 45),
                 zoom_range = (0.95, 1.05), #(0.9, 1.05), 
                 scale_int = (0, 255),
                 erosion_kernel_size = 11,
                 int_aug_offset = None, #(-0.2, 0.2), #
                 int_aug_expansion = None, # (0.75, 1.2), #
                 samples_per_epoch = 2500,
                 convolve_egg_flag_kernel = [0, 0.15, 1., 0.6, 0.25]#[0, 0.15, 1., 0.3, 0.15]
                 ):
        
        
        self.root_dir = root_dir
        self.positive_prob = positive_prob
        self.snippet_size = snippet_size
        self.max_offset = max_offset
        self.max_offset_per_frame = max_offset_per_frame
        self.motion_blur_range = motion_blur_range
        self.zoom_range = zoom_range
        self.scale_int = scale_int
        self.erosion_kernel_size = erosion_kernel_size
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        self.samples_per_epoch = samples_per_epoch
        self.convolve_egg_flag_kernel = convolve_egg_flag_kernel
         
        fnames = [x.resolve() for x in root_dir.rglob('*.hdf5')]
        
        
        data = defaultdict(list)
        for fname in tqdm.tqdm(fnames):
            with tables.File(fname, 'r') as fid:
                snippets = fid.get_node('/snippets')[:]
                egg_flags = fid.get_node('/egg_flags')[:]
                
                snippets = snippets.astype(np.float32)
                egg_flags = egg_flags.astype(np.float32)
                #egg_flags = egg_flags.astype(np.int)
            
            for snippet, egg_flag in zip(snippets, egg_flags):
                k = 'positive' if np.any(egg_flag) else 'negative'
                data[k].append((snippet, egg_flag))
        self.data = data
        
        transforms = [#RandomTemporalCrop(self.snippet_size),
                      CenterTemporalCrop(snippet_size = self.snippet_size),
                      RandomErosion(kernel_size = erosion_kernel_size),
                      RandomFillBackground(),
                      RandomMotionBlur(displacement_range = self.motion_blur_range),
                      RandomHorizontalFlip(), 
                      RandomVerticalFlip(),
                      RandomOffset(max_offset = self.max_offset),
                      RandomOffsetPerFrame(max_offset = self.max_offset_per_frame),
                      RandomZoom(zoom_range = self.zoom_range), 
                      NormalizeIntensity(scale = self.scale_int),
                      RandomIntensityOffset(offset_range = self.int_aug_offset),
                      RandomIntensityExpansion(expansion_range = self.int_aug_expansion)
                      ]
        
        if self.convolve_egg_flag_kernel is not None:
            transforms.append(ConvolveEggFlag(self.convolve_egg_flag_kernel))
        
        self.transforms = Compose(transforms)
        
    def __getitem__(self, ind):
        k = 'positive' if random.random() < self.positive_prob else 'negative'
        snippet, egg_laying_vector = random.choice(self.data[k])
        snippet, egg_laying_vector =  self.transforms(snippet, egg_laying_vector)
        
        return snippet, egg_laying_vector
    
    def __len__(self):
        return self.samples_per_epoch
#%%
class SnippetsFullFlow():
    def __init__(self, 
                 root_dir,
                 scale_int = (0, 255)
                 ):
        
        self.root_dir = root_dir
        self.scale_int = scale_int
        
        fnames = [x.resolve() for x in root_dir.rglob('*.hdf5')]
        data = []
        for fname in tqdm.tqdm(fnames, 'Preloading data...'):
            with tables.File(fname, 'r') as fid:
                snippets = fid.get_node('/snippets')[:]
                egg_flags = fid.get_node('/egg_flags')[:]
                
                snippets = snippets.astype(np.float32)
                egg_flags = egg_flags.astype(np.float32)
            
            for (snippet, egg_flag) in zip(snippets, egg_flags):
                data.append((snippet, egg_flag))
            
        self.data = data
        
        transforms = [NormalizeIntensity(self.scale_int)]
        self.transforms = Compose(transforms)
        
    def __getitem__(self, ind):
        return self.transforms(*self.data[ind])
    
    def __len__(self):
        return len(self.data)
    #%%
if __name__ == '__main__':
    from matplotlib.pylab import plt
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/v4/test'
    root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/data/mixed-setups/test'
    
    #root_dir = Path('/Users/avelinojaver/Desktop/syngenta_sequences/final/test')
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/eggs/snippets/mixed-setups/test/')
    flow_train_argkws = dict(
                 snippet_size = 7,
                 max_offset = 12,
                 max_offset_per_frame = 5,
                 zoom_range = (0.9, 1.1),
                 scale_int = (0, 255),
                 erosion_kernel_size = 5,
                 int_aug_offset =(-0.1, 0.1), 
                 int_aug_expansion = (0.9, 1.1),
                 convolve_egg_flag_kernel = [0.1, 0.4, 1., 0.4, 0.1]
                 )
    
    
    #gen = SnippetsFullFlow(root_dir)
    gen = SnippetsRandomFlow(root_dir, **flow_train_argkws)
    #%%
    for ii in tqdm.tqdm(range(10)):
        snippets, is_egg_laying  = gen[ii]
        
        fig, axs = plt.subplots(1, len(snippets), figsize = (25, 5), sharex = True, sharey = True)
        for ax, ss, flag in zip(axs, snippets, is_egg_laying):
            ax.imshow(ss, cmap = 'gray')
            ax.set_title(flag)
        
        
    