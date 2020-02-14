#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:15:41 2019

@author: avelinojaver
"""

import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[3] / 'WT2' 
sys.path.append(str(dname))

from process.process_files_all import load_model, model2gpus

import pandas as pd
import numpy as np
import multiprocessing as mp
import tqdm
import torch
import torch.nn.functional as F
import random

def read_rois_proc(mask_file, trajectories_data, queue, roi_size, snippet_length, snippet_overlap):
    
    half_roi = roi_size//2
    
    def _correct_lims(c, c_lim):
        cl, cr = c - half_roi, c + half_roi
        if cl < 0:
            cl_pad = -cl
            cl = 0
        else:
            cl_pad = 0
        
        if cr > c_lim:
            cr_pad = cr - c_lim
            cr = c_lim
        else:
            cr_pad = 0
        return (cl, cr), (cl_pad, cr_pad)
    

    worm_data = {}
    with pd.HDFStore(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_h, img_w = masks.shape
        
        
        last_worm_frame = trajectories_data[['worm_index_joined', 'frame_number']].groupby('worm_index_joined').max()['frame_number']
        
        
        bn = mask_file.stem
        for frame_number, frame_data in tqdm.tqdm(trajectories_data.groupby('frame_number'), desc = bn):
            img = masks[frame_number]
                            
            for _, row in frame_data.iterrows():
                w_index = int(row['worm_index_joined'])
                x = int(row['coord_x'])
                y = int(row['coord_y'])
                
                
                xlims, xpads = _correct_lims(x, img_w)
                ylims, ypads = _correct_lims(y, img_h)
                
                roi = img[ ylims[0]:ylims[1], xlims[0]:xlims[1]]
                
                if roi.shape != (roi_size, roi_size):
                    roi = np.pad(roi, (ypads, xpads))
                    assert roi.shape == (roi_size, roi_size)
                
                if not w_index in worm_data:
                    worm_data[w_index] = []
                worm_data[w_index].append((frame_number, x, y, roi))
                if (len(worm_data[w_index]) >= snippet_length) or (frame_number >= last_worm_frame[w_index]):
                    dat = worm_data[w_index]
                    queue.put((w_index, dat[:snippet_length])) # make sure the passed date is at most the snippet length
                    
                    if frame_number < last_worm_frame[w_index]:
                        #let's start adding overlap with the next batch
                        worm_data[w_index] = dat[-snippet_overlap:] 
                    else:
                        del worm_data[w_index]
                    
                
    for w_index, val in worm_data.items():
        queue.put((w_index, val))
    queue.put(None)


@torch.no_grad()
def get_prediction(batch, model, device,  snippet_shape):
    X = batch.to(device)
    
    if batch.shape[2:] != (snippet_shape, snippet_shape):
        X = F.interpolate(X, snippet_shape)
    
    predictions = model(X)
    
    
    predictions = torch.sigmoid(predictions)
    predictions = predictions.squeeze(-1)
    predictions = predictions.detach().cpu().numpy()
    
    return predictions



@torch.no_grad()
def get_prediction_with_aug(batch, model, device,  snippet_shape):
    #%%
    X = batch.to(device)
    
    if batch.shape[2:] != (snippet_shape, snippet_shape):
        X = F.interpolate(X, snippet_shape)
    
    X_with_aug = torch.cat((X, X.flip(-1), X.flip(-2), X.transpose(-2, -1)))
    predictions = model(X_with_aug)
    predictions = torch.sigmoid(predictions)
    predictions = predictions.squeeze(-1)
    predictions = predictions.detach().cpu().numpy()
    
    return predictions

def main():
    
    image_roi_size = 96
    snippet_shape = 96
    snippet_offset = 3
    batch_size = 1
    
    # snippet_length = 640
    # is_augment = False
    
    snippet_length = 160
    is_augment = True
    
    assert batch_size == 1
    
    cuda_id = 0
    model_path_root = None
    
    #model_base = 'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10' 
    
    #model_base = 'WT+mixed-setups_unet-v4+R_20200131_171115_adam-_lr5e-05_wd0.0_batch32'
    #model_base = 'WT+mixed-setups_unet-v4_20200131_171116_adam-_lr0.0001_wd0.0_batch32'
    
    #model_base = 'WT+mixed-setups_unet-v4+R_BCEp2_20200205_120934_adam-_lr5e-05_wd0.0_batch32'
    #model_base = 'WT+mixed-setups_unet-v4+R_BCE_20200207_181629_adam-_lr5e-05_wd0.0_batch32'
    
    model_base = 'WT+mixed-setups-v2_unet-v4+R_BCE_20200210_002129_adam-_lr5e-05_wd0.0_batch32'
    #%%
    data_root_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/'
    data_type = 'syngenta'
    
    #data_root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/'
    #data_type = 'CeNDR'
    
    mask_files = [x for x in Path(data_root_dir).rglob('*.hdf5') if not x.name.startswith('.')]
    #mask_files = [x for x in Path(data_root_dir).rglob('N2_worms10_No_Compound_0_Set4_Pos5_Ch5_18072017_192541.hdf5') if not x.name.startswith('.')]
    
    feats_files = [Path(str(x).replace('/MaskedVideos/', '/Results/')[:-5] + '_featuresN.hdf5') for x in mask_files]
    files2check = [d for d in zip(mask_files, feats_files) if d[1].exists()]

    #postfix = 'Syngenta_Agar_Screening_003_210717/N2_worms10_CSAA062324_10_Set5_Pos4_Ch2_21072017_213453.hdf5'
    #mask_file = data_root_dir / 'MaskedVideos' / postfix
    #features_file = data_root_dir / 'Results' / (postfix[:-5] + '_featuresN.hdf5')
    
    #%% 
    if model_path_root is None:
        model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    model_path_root = Path(model_path_root)
    assert model_path_root.exists()
    
    model_path = model_path_root / model_base / 'model_best.pth.tar'
    
    
    bb = 'AUG_' + model_base if is_augment else model_base 
    root_save_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions_v2' / data_type  / bb 
    root_save_dir.mkdir(exist_ok = True, parents = True)
    
    model, model_args = load_model(model_path)
    model, device, batch_size = model2gpus(model, cuda_id)
    
    random.shuffle(files2check)
    for mask_file, feats_file in tqdm.tqdm(files2check):
    
        bn = mask_file.stem
        save_name = root_save_dir / (bn + '_eggs.csv')
        if save_name.exists():
            continue
        
        with pd.HDFStore(feats_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        
        pqueue = mp.Queue(5)
        reader_p = mp.Process(target = read_rois_proc, 
                              args= (mask_file, trajectories_data, pqueue, image_roi_size, snippet_length, 2*snippet_offset)
                              )
        reader_p.daemon = True
        reader_p.start()        # Launch reader_proc() as a separate python process
    
        predictions = []
        while True:
            dat = pqueue.get()
            if dat is None:
                break
            
            worm_index, worm_data = dat
            frame_numbers, cx, cy, roi_sequence = map(np.array, zip(*worm_data))
            
            batch = torch.from_numpy(roi_sequence[None])
            batch = batch.float()/255
            
            if is_augment:
                egg_flags = get_prediction_with_aug(batch, model, device,  snippet_shape)
                assert egg_flags.shape[0] == 4
            else:
                egg_flags = get_prediction(batch, model, device,  snippet_shape)
            egg_probs = np.mean(egg_flags, axis = 0)
            egg_err = np.std(egg_flags, axis = 0)
            
            frame_numbers, cx, cy, egg_probs, egg_err = [x[snippet_offset:-snippet_offset] for x in (frame_numbers, cx, cy, egg_probs, egg_err)]
            w_inds = np.full(len(frame_numbers), worm_index)
            predictions.append((w_inds, frame_numbers, cx, cy, egg_probs, egg_err))
            
        dat = list(zip(*map(np.concatenate, zip(*predictions))))
        df = pd.DataFrame(dat, columns = ['worm_index', 'frames', 'x', 'y', 'eggs_prob', 'egg_err'])
        df.to_csv(save_name, index = False, float_format='%.2f')
    
#if __name__ == '__main__':
def _main_debug():
    #model_base = 'WT+mixed-setups_unet-v4+R_20200131_171115_adam-_lr5e-05_wd0.0_batch32'
    #model_base = 'WT+mixed-setups_unet-v4+R_BCEp2_20200205_120934_adam-_lr5e-05_wd0.0_batch32'
    model_base = 'WT+mixed-setups_unet-v4+R_BCE_20200207_181629_adam-_lr5e-05_wd0.0_batch32'
    
    data_type = 'syngenta'
    model_path_root = Path.home() / 'workspace/WormData/egg_laying/single_worm/results/'
    assert model_path_root.exists()
    
    model_path = model_path_root / model_base / 'model_best.pth.tar'
    
    model, model_args = load_model(model_path)
   
    image_roi_size = 96
    snippet_shape = 96
    snippet_length = 128
    snippet_offset = 3
    
    #mask_file, feats_file = files2check[0]
    mask_file = Path('/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/Syngenta_Agar_Screening_002_180717/N2_worms10_No_Compound_0_Set4_Pos5_Ch5_18072017_192541.hdf5')
    feats_file = Path('/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/Results/Syngenta_Agar_Screening_002_180717/N2_worms10_No_Compound_0_Set4_Pos5_Ch5_18072017_192541_featuresN.hdf5')
    
    
    with pd.HDFStore(feats_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    pqueue = mp.Queue(5)
    reader_p = mp.Process(target = read_rois_proc, 
                          args= (mask_file, trajectories_data, pqueue, image_roi_size, snippet_length, 2*snippet_offset)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    #%%
    predictions = []
    predictions_aug = []
    ROIs = []
    device = 'cpu'
    for ii in range(800):
        worm_index, worm_data = pqueue.get()
        if worm_index == 9:
            
            frame_numbers, cx, cy, roi_sequence = map(np.array, zip(*worm_data))
            if (frame_numbers < 2500).any():
                continue
            
            
            #if (frame_numbers > 600).any() & (frame_numbers < 700).any():
            #TODO add support for larger batches (multi gpu)
            batch = torch.from_numpy(roi_sequence[None])
            batch = batch.float()/255
            
            #egg_flags_aug = get_prediction_with_aug(batch, model, device,  snippet_shape)
            egg_flags_aug = get_prediction(batch, model, device,  snippet_shape)
            #assert egg_flags_aug.shape[0] == 4
            egg_probs = np.mean(egg_flags_aug, axis = 0)
            egg_err = np.std(egg_flags_aug, axis = 0)
            
            frame_numbers, cx, cy, egg_probs, egg_err = [x[snippet_offset:-snippet_offset] for x in (frame_numbers, cx, cy, egg_probs, egg_err)]
            w_inds = np.full(len(frame_numbers), worm_index)
            predictions.append((w_inds, frame_numbers, cx, cy, egg_probs, egg_err))
            predictions_aug.append(egg_flags_aug)
            ROIs.append(batch[0, snippet_offset:-snippet_offset].numpy())
        #%%
    dat = list(zip(*map(np.concatenate, zip(*predictions))))
    df = pd.DataFrame(dat, columns = ['worm_index', 'frames', 'x', 'y', 'eggs_prob', 'eggs_err'])
    
    
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(df['frames'], df['eggs_prob'])
    
    
    R = np.concatenate(ROIs)
    ind = np.argmax(df['eggs_prob'].values)
    
    
    fig, axs = plt.subplots(1, 3, sharex = True, sharey  = True)
    axs[0].imshow(R[ind-2])
    axs[1].imshow(R[ind])
    axs[2].imshow(R[ind+2])
    #%%
    bb = R[None, ind - 3 : ind + 4]
    bb = torch.from_numpy(bb)
    flags = get_prediction(bb, model, device,  snippet_shape)
   
    #%%
if __name__ == '__main__':
       main()

    