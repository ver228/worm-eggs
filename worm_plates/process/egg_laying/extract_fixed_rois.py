#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:07:48 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import tables
from tqdm import tqdm
from collections import defaultdict
import torch
import multiprocessing as mp
import random

from cell_localization.utils import get_device
from cell_localization.models import get_model

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



@torch.no_grad()
def get_predictions_with_aug(model, X_ori):
    bkp_flag = model.return_belive_maps
    model.return_belive_maps = True
    
    for ii in range(4):
        if ii == 0:
            transform_func = lambda x : x
        elif ii == 1:
            transform_func = lambda x : x.flip(-1)
        elif ii == 1:
            transform_func = lambda x : x.flip(-2)
        else:
            transform_func = lambda x : x.transpose(-1, -2)
            
        X = transform_func(X_ori)
        _, b_maps = model(X)
        
        if isinstance(b_maps, tuple):
            b_maps = b_maps[0]
        
        b_maps = transform_func(b_maps)
        
        
        if ii == 0:
            belive_maps = b_maps
        else:
            belive_maps += b_maps
    
    belive_maps /= 4
    
    outs = model.nms(belive_maps)
    predictions = []
    for coordinates, labels, scores_abs, scores_rel in outs:
        res = dict(
                    coordinates = coordinates,
                    labels = labels,
                    scores_abs = scores_abs,
                    scores_rel = scores_rel,
                    
                    )
        predictions.append(res)

    model.return_belive_maps = bkp_flag
    return predictions

@torch.no_grad()
def get_eggs_full_frames(fname, model_eggs, augment = False):
    device = next(model_eggs.parameters()).device #use the same device as the model
    with tables.File(fname, 'r') as fid:
        full_frames = fid.get_node('/full_data')
        full_frame_interval = full_frames._v_attrs['save_interval']
        img_shape = full_frames.shape[1:]
        
        imgs = full_frames[:].astype(np.float32)/255
    
    
    
    eggs_full_frames = []
    for ii, img in enumerate(imgs):
        #I am passing one image at a time. This might cause problems down the line if the image is to big to fit in the gpu 
        #or be inefficient if the image is too small
        X = torch.from_numpy(img[None, None])
        X = X.to(device)
        
        if augment:
            predictions = get_predictions_with_aug(model_eggs, X)
        else:
            predictions = model_eggs(X)
        
        frame_number = ii
        pred = predictions[0]
        
        
        res = [pred[x].detach().cpu().numpy() for x in ['coordinates', 'scores_abs', 'scores_rel']]
    
        res = [x[:, None] if x.ndim == 1 else x for x in res]
        res = np.concatenate(res, axis=1)
        eggs_full_frames += [(frame_number, *cc) for cc in zip(*res.T)]
            
    eggs_full_frames = pd.DataFrame(eggs_full_frames, columns = ['frame_number', 'x', 'y', 'score_abs', 'score_rel'])
    
    return eggs_full_frames, img_shape, full_frame_interval

def link_eggs_full_frames(df, max_dist = 25):
    eggs_l = []
    for frame_number, frame_data in df.groupby('frame_number'):
        rec = frame_data.to_records(index = False)
        
        if not eggs_l:
            eggs_l = [[x] for x in rec]
            prev_index = list(range(len(eggs_l)))
        else:
            prev_coords = [eggs_l[i][-1] for i in prev_index]
            prev_coords = np.array([(x['x'], x['y']) for x in prev_coords])
            
            curr_coords = np.stack((rec['x'], rec['y'])).T
            dists = cdist(prev_coords, curr_coords)
            
            #get the closest index to a given index
            closest_match = np.argmin(dists, axis = 1)
            
            
            matched_pairs = [(prev_index[ii], i_next) for ii, i_next in enumerate(closest_match) if dists[ii, i_next] < max_dist]
            
            next_index = []
            for i_prev, i_next in matched_pairs:
                eggs_l[i_prev].append(rec[i_next])
                next_index.append(i_prev)
                
            #add new blinkx
            matched_rows = set([x[1] for x in matched_pairs])
            inds2add = set(range(len(rec))) - matched_rows
            
            
            nn = len(eggs_l)
            eggs_l += [[rec[i]] for i in inds2add]
            
            next_index += range(nn, nn + len(inds2add))
            
            prev_index = next_index
    return eggs_l

def read_blobs_bboxes(feats_file):
    with pd.HDFStore(feats_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        blob_features = fid['/blob_features']
        microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
        
    w_half = blob_features['box_width']/2
    l_half = blob_features['box_length']/2
    
    bbox = {'frame_number' : trajectories_data['frame_number'],
            'xmin' : blob_features['coord_x'] - w_half,
            'xmax' : blob_features['coord_x'] + w_half,
            'ymin' : blob_features['coord_y'] - l_half,
            'ymax' : blob_features['coord_y'] + l_half
         }
    
    
    bbox = pd.DataFrame(bbox)
    bbox[['xmin', 'ymin', 'xmax', 'ymax']] /= microns_per_pixel
    
    return bbox

def get_candidate_bboxes(eggs2check, blop_bboxes, roi_size, img_shape, full_frame_interval):
    roi_area = roi_size**2
    half_roi = 96//2
    
    assert all([x>=roi_size for x in img_shape])
    
    egg_rois2check = defaultdict(list)
    
    xl_lim = img_shape[0] - roi_size
    yl_lim = img_shape[1] - roi_size
    for egg in eggs2check:
        xl = max(0, int(egg['x'] - half_roi))
        xl = min(xl, xl_lim)
        yl = max(0, int(egg['y'] - half_roi))
        yl = min(yl, yl_lim)
        
        egg_roi = np.array((xl, yl, xl + roi_size, yl + roi_size))
        
        assert (egg_roi[0] >= 0) & (egg_roi[1] >= 0) & (egg_roi[2] <= img_shape[0]) & (egg_roi[3] <= img_shape[1])
        
        
        t0, t1 = (egg['frame_number'] - 1)*full_frame_interval, egg['frame_number']*full_frame_interval
        
        valid = (blop_bboxes['frame_number'] > t0) & (blop_bboxes['frame_number'] < t1)
        boxes2check = blop_bboxes.loc[valid, ['xmin', 'ymin', 'xmax', 'ymax']].values.T
        
    
        c_min = np.maximum(egg_roi[:2, None], boxes2check[:2])
        c_max = np.minimum(egg_roi[2:, None], boxes2check[2:])
    
    
        w = np.maximum(0.0, c_max[0] - c_min[0] + 1)
        h = np.maximum(0.0, c_max[1] - c_min[1] + 1)
        inter = w * h/roi_area
        
        frames2read = blop_bboxes.loc[valid, 'frame_number'][inter>0.1].values
    
        for f in frames2read:
            egg_rois2check[f].append(egg_roi)
    return egg_rois2check

def candidate_reader_proc(mask_file, egg_rois2check, batch_size, queue):
    batch = []
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        for frame_number, rois2check in tqdm(egg_rois2check.items()):
            try:
                img = masks[frame_number]
            except:
                continue

            for xmin, ymin, xmax, ymax in rois2check:
                roi = img[ymin:ymax, xmin:xmax]
                
                batch.append((frame_number, (xmin, ymin), roi))
                if len(batch) >= batch_size:
                    batch = list(map(np.array, zip(*batch)))
                    queue.put(batch)
                    batch = []
    if batch:
        batch = list(map(np.array, zip(*batch)))
        queue.put(batch)
    
    queue.put(None)
        
        


@torch.no_grad()
def get_eggs_in_masks(model_eggs, mask_file, egg_rois2check, batch_size, augment = False):
    device = next(model_eggs.parameters()).device #use the same device as the model
    
    pqueue = mp.Queue()
    reader_p = mp.Process(target = candidate_reader_proc, 
                          args= (mask_file, egg_rois2check, batch_size, pqueue)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    
    preds_l = []
    while True:
        batch = pqueue.get()
        if batch is None:
            break
        
        frame_numbers, corners, rois = batch
        X = rois[:, None].astype(np.float32)/255
        X = torch.from_numpy(X)
        X = X.to(device)
        
        if augment:
            predictions = get_predictions_with_aug(model_eggs, X)
        else:
            predictions = model_eggs(X)
            
        for frame_number, corner, preds in zip(frame_numbers, corners, predictions):
            res = [preds[x].detach().cpu().numpy() for x in ['coordinates', 'scores_abs', 'scores_rel']]
            res[0] = res[0] + corner[None, :]
            
            res = [x[:, None] if x.ndim == 1 else x for x in res]
            res = np.concatenate(res, axis=1)
            
            preds_l += [(frame_number, *cc) for cc in zip(*res.T)]
        
    eggs_mask_frames = pd.DataFrame(preds_l, columns = ['frame_number', 'x', 'y', 'score_abs', 'score_rel'])
    return eggs_mask_frames

def get_egg_laying_events(eggs2check, eggs_by_frame, full_frame_interval, max_dist = 5):
    all_trajs = []
    for seed in eggs2check:
        t0, t1 =  (seed['frame_number'] -1)*full_frame_interval, seed['frame_number']*full_frame_interval
        
        
        traj = [(t1, seed['x'], seed['y'])] 
        for frame in range(t1-1, t0, -1):
            if not frame in eggs_by_frame:
                continue
            
            eggs_in_frame = eggs_by_frame[frame]
            dx = traj[-1][1] - eggs_in_frame['x']
            dy = traj[-1][2] - eggs_in_frame['y']
            
            r = np.sqrt(dx*dx + dy*dy)
            good = r < max_dist
            if np.any(good):
                ind = np.argmin(r)
                
                row = eggs_in_frame[ind]
                traj.append((frame, row['x'], row['y']))
        all_trajs.append(traj)
        
    egg_laying_events = [(*x[-1], *x[0]) for x in all_trajs]
    if not egg_laying_events:
        egg_laying_events =  np.zeros((0, 6))
    
    egg_laying_events = pd.DataFrame(egg_laying_events, columns = ['frames', 'x', 'y', 'src_frame', 'src_x', 'src_y'])
    
    
    return egg_laying_events

def extract_eggs_from_file(
    mask_file, 
    feats_file,
    model,
    save_dir,
    
    max_dist_btw_full_frames = 50,
    max_dist_btw_mask_frames = 5,
    
    roi_size = 96,
    batch_size = 128,
    augment = False,
    only_full_eggs = False
    ):
    
    save_name_mask = save_dir / (mask_file.stem + '_eggs_mask.csv')
    save_name_full = save_dir / (mask_file.stem + '_eggs_full.csv')
    save_name_events = save_dir / (mask_file.stem + '_eggs_events.csv')
    
    if save_name_events.exists():
        return
    
    if save_name_full.exists() and not only_full_eggs:
        tqdm.write('Reading eggs from full frames...')
        eggs_full_frames = pd.read_csv(save_name_full)
        
        with tables.File(mask_file, 'r') as fid:
            full_frames = fid.get_node('/full_data')
            full_frame_interval = full_frames._v_attrs['save_interval']
            img_shape = full_frames.shape[1:]
            
    else:
        if not save_name_full.exists():
            tqdm.write('Extracting eggs from full frames...')
            eggs_full_frames, img_shape, full_frame_interval = get_eggs_full_frames(mask_file, model, augment = augment)
            eggs_full_frames.to_csv(save_name_full, index = False)
            
    if not only_full_eggs:
        eggs_trajectories = link_eggs_full_frames(eggs_full_frames, max_dist = max_dist_btw_full_frames)
        eggs2check = [x[0] for x in eggs_trajectories if x[0]['frame_number'] > 0]
        
        
        blop_bboxes = read_blobs_bboxes(feats_file)
        egg_rois2check = get_candidate_bboxes(eggs2check, blop_bboxes, roi_size, img_shape, full_frame_interval)
        
        eggs_mask_frames = get_eggs_in_masks(model, mask_file, egg_rois2check, batch_size = batch_size, augment = augment)
        eggs_by_frame = {frame_number:frame_data.to_records(index=False) for frame_number, frame_data in eggs_mask_frames.groupby('frame_number')}
        
        egg_laying_events = get_egg_laying_events(eggs2check, eggs_by_frame, full_frame_interval, max_dist = max_dist_btw_mask_frames)
        
        eggs_mask_frames.to_csv(save_name_mask, index = False)
        egg_laying_events.to_csv(save_name_events, index = False)
    


def main(
    model_path = None,
    nms_threshold_rel = 0.25,
    cuda_id = 0,
    augment = True,
    #root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos',
    data_root_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/',
    save_dir_root = Path.home() / 'workspace/WormData/egg_laying/plates/predictions/syngenta',
    only_full_eggs = False
    ):
    
    if model_path is None:
        bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
        model_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/'/ bn.partition('+F')[0] / bn / 'model_best.pth.tar'
    else:
        model_path = Path(model_path)
        assert model_path.exists() 
        bn = model_path.parent.name
        
    model_args = dict(nms_threshold_abs = 0.,
                            nms_threshold_rel = nms_threshold_rel,
                            pad_mode = 'reflect'
                            )
    
    device = get_device(cuda_id)
    model, epoch = load_model(model_path, **model_args)
    model = model.to(device)
    
    
    save_subdir = f'AUG_{bn}' if augment else bn
    save_dir = Path(save_dir_root) / save_subdir
    save_dir.mkdir(parents = True, exist_ok = True)
    
    mask_files = [x for x in Path(data_root_dir).rglob('*.hdf5') if not x.name.startswith('.')]
    feats_files = [Path(str(x).replace('/MaskedVideos/', '/Results/')[:-5] + '_featuresN.hdf5') for x in mask_files]
    
    files2check = [d for d in zip(mask_files, feats_files) if d[1].exists()]
    
    
    random.shuffle(files2check)
    #mask_file = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos/CeNDR_Set1_020617/N2_worms10_food1-10_Set3_Pos4_Ch3_02062017_123419.hdf5'
    for mask_file, feats_file in tqdm(files2check):
        
        
        extract_eggs_from_file(mask_file, feats_file, model, save_dir, augment = augment, only_full_eggs = only_full_eggs)

if __name__ == '__main__':
    import fire
    fire.Fire(main)