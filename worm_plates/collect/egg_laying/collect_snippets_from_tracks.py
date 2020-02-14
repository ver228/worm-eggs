#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:59:05 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
import tqdm
import tables
import numpy as np
import cv2
import pickle

def _correct_lims(c, c_lim, roi_size):
    half_roi = roi_size//2
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


def _process_row(dat):
    f_bn, mask_file, feats_file, event_file, thresh = dat
    event_df = pd.read_csv(event_file)
    if 'eggs_prob' in event_df:
        event_df = event_df[event_df['eggs_prob'] >= thresh]
    
    with tables.File(feats_file) as fid:
        trajectories_data = fid.get_node('/trajectories_data')[:]
    
    
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_w, img_h = masks.shape
        
        last_frame = masks.shape[0] - 1
        
        worm_data_d = {}
        for frame, events_in_frame in event_df.groupby('frames'):
            snippet_ini, snippet_fin = (frame - s_half_size), (frame + s_half_size)
            snippet_ini = max(0, snippet_ini)
            snippet_fin = min(snippet_fin, last_frame)
            masks_snippet = masks[snippet_ini:snippet_fin + 1]
            
            
            
            for irow, event in events_in_frame.iterrows():
                w_ind = int(event['worm_index'])
                if w_ind in worm_data_d:
                    worm_data = worm_data_d[w_ind]
                else:
                    worm_data = trajectories_data[(trajectories_data['worm_index_joined'] == w_ind)]
                    worm_data_d[w_ind] = worm_data
                
                
                snippet_coords = worm_data[(worm_data['frame_number'] >= snippet_ini) & (worm_data['frame_number'] <= snippet_fin)]
                try:
                    if len(snippet_coords) != len(masks_snippet):
                        raise ValueError
                    snippet_coords = sorted(snippet_coords, key = lambda x : x['frame_number'])
                    snippet_r = []
                    for ss, img in zip(snippet_coords, masks_snippet):
                        xlims, xpads = _correct_lims(int(ss['coord_x']), img_w, roi_size)
                        ylims, ypads = _correct_lims(int(ss['coord_y']), img_h, roi_size)
                        roi = img[ylims[0]:ylims[1], xlims[0]:xlims[1]]
                        snippet_r.append(roi)
                    snippet_r = np.stack(snippet_r)
                except ValueError:
                    xlims, xpads = _correct_lims(int(event['x']), img_w, roi_size)
                    ylims, ypads = _correct_lims(int(event['y']), img_h, roi_size)
                    snippet_r = masks_snippet[:, ylims[0]:ylims[1], xlims[0]:xlims[1]]
                
                    
                    
                if not np.any(snippet_r):
                    continue
                
                rr = roi_centre_pad
                ss = np.concatenate([x[rr:-rr, rr:-rr] for x in snippet_r], axis=1)
                cv2.imwrite(str(save_images_dir / f'{f_bn}_{irow}_snippet.png'), ss)
                
                with open(save_snippets_dir / f'{f_bn}_{irow}_snippet.pickle', 'wb') as fid:
                    pickle.dump(snippet_r, fid)
                
                #plt.figure()
                #plt.imshow(ss)
    


    


if __name__ == '__main__':
    import multiprocessing as mp
    
    snippet_size = 7
    s_half_size = snippet_size//2
    thresh = 0.7#0.85
    is_augment = False
    
    roi_size = 96
    r_half_size = roi_size//2
    roi_centre_pad = roi_size//4
    #bn = 'AUG_worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    #bn = 'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
    #bn = 'WT+mixed-setups_unet-v4_20200131_171116_adam-_lr0.0001_wd0.0_batch32'
    #bn = 'WT+mixed-setups_unet-v4+R_20200131_171115_adam-_lr5e-05_wd0.0_batch32'
    #bn = 'WT+mixed-setups_unet-v4+R_BCEp2_20200205_120934_adam-_lr5e-05_wd0.0_batch32'
    #bn = 'WT+mixed-setups_unet-v4+R_BCE_20200207_181629_adam-_lr5e-05_wd0.0_batch32'
    
    
    #bn = 'WT+mixed-setups-v2_unet-v4+R_BCE_20200210_002129_adam-_lr5e-05_wd0.0_batch32'
    bn = 'AUG_WT+mixed-setups-v2_unet-v4+R_BCE_20200210_002129_adam-_lr5e-05_wd0.0_batch32'
    
    
    data_type = 'CeNDR'
    data_dir = Path.home() / 'workspace/WormData/screenings/CeNDR/MaskedVideos'
    
    # data_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos'
    # data_type = 'syngenta'
    
    events_dir = Path.home() / 'workspace/WormData/egg_laying/plates/predictions_v2/'  /  data_type / bn
    save_root_dir = Path.home() / 'workspace/WormData/egg_laying/plates/annotations2check' / data_type / bn
    
    save_images_dir = save_root_dir / 'images'
    save_snippets_dir = save_root_dir / 'snippets'
    
    save_images_dir.mkdir(parents = True, exist_ok = True)
    save_snippets_dir.mkdir(parents = True, exist_ok = True)
    
    assert data_dir.exists()
    assert events_dir.exists()
    
    mask_files = data_dir.rglob('*.hdf5')
    feature_files = (data_dir.parent / 'Results').rglob('*_featuresN.hdf5')
    
    
    #events_postfix = '_eggs_events.csv'
    #events_files = [x for x in events_dir.rglob('*' + events_postfix) if  not x.name.startswith('.')]
    
    events_postfix = '_eggs.csv'
    events_files = [x for x in events_dir.rglob('*' + events_postfix) if  not x.name.startswith('.')]
    
    mask_files_d = {x.stem:x for x in mask_files}
    feature_files_d = {x.name[:-15]:x for x in feature_files}
    events_files_d = {x.name[:-len(events_postfix)] : x for x in events_files}
    
    
    #%%
    files2process = [(f_bn, mask_files_d[f_bn], feature_files_d[f_bn], event_file, thresh) for f_bn, event_file in events_files_d.items() if f_bn in mask_files_d]
    
    #for row in tqdm.tqdm(files2process):
    #    _process_row(row)
        
    
    with mp.Pool(8) as p:
      r = list(tqdm.tqdm(p.imap(_process_row, files2process), total=len(files2process)))
        
            
         
        