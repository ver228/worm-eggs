#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:54:11 2019

@author: avelinojaver
"""

import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt

import pandas as pd
from scipy.signal import find_peaks

import sys

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))

#


if __name__ == '__main__':
    
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/test/model_best/'
    root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/'
    
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/_old/199/'
    #root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm/predictions/_old/aug-199/'

    th2check = np.arange(0.0, 1, 0.025)
    
    #SAVE_ERRORS = False
    th2plot = None
    
    valid_size = ((150,-150), (230,-230))
    models2check = sorted([x.name for x in root_dir.glob('*/') if not x.name.startswith('_')])
    
    #models2check = ['WT+v3_unet-v3_20191024_173911_adam-_lr0.0001_wd0.0_batch4']; th2plot = 0.8
    #models2check = ['WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8']; th2plot = 0.6
    
    
    #th2plot = 0.8
    # models2check = [
    #              'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4',
    #              'WT+v3+hard-neg_unet-v4+R_20191028_205224_adam-_lr0.0001_wd0.0_batch8',
    #              'WT+v3+hard-neg_unet-v4+R_20191028_212208_adam-_lr0.0001_wd0.0_batch8',
    #              'WT+v4_unet-v4_20191029_153801_adam-_lr0.0001_wd0.0_batch10_frame-shift+kernel-simple',
    #              'WT+v4_unet-v4_20191029_181002_adam-_lr0.0001_wd0.0_batch10_frame-shift',
    #              'WT+v4_unet-v4_20191030_143735_adam-_lr0.0001_wd0.0_batch10_intensity+kernel-simple',
    #              'WT+v4_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
    #              'WT+v2+hard-neg-2_unet-v4_20191031_164920_adam-_lr0.0001_wd0.0_batch10',
    #              'WT+v4_unet-v4_20191101_160320_adam-_lr0.0001_wd0.0_batch10',
    #              'WT+v4_unet-v4+hardaug_20191031_214855_adam-_lr0.0001_wd0.0_batch10',
                 
    #              'WT+v4_unet-v4+R_20191102_175649_adam-_lr0.0001_wd0.0_batch10',
    #              'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
    #             ]
    
    models2check = [
                  #'WT+v2+hard-neg-2_unet-v3_20190907_135706_adam-_lr0.0001_wd0.0_batch4',
    
                  'WT+v4_unet-v4+R_20191101_160208_adam-_lr1e-05_wd0.0_batch10'
                ]
    
    if th2plot is not None:
        ith2plot = np.abs(th2check - th2plot).argmin()
    
    all_metrics = {}
    
    #%%
    
    if th2plot is not None:
        features_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
        ts_files = [x for x in features_dir.rglob('*.hdf5') if not (x.name.endswith('_featuresN.hdf5') or x.name.endswith('_interpolated25.hdf5'))]
        ts_files = {x.stem:x for x in ts_files}
    
    #%%
    ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events_231019.tsv'
    df_true = []
    with open(ANNOTATIONS_FILE) as fid:
        for row in fid.read().split('\n'):
            if row:
                dd = row.split('\t')
                bn = dd[0]
                for event in dd[1:]:
                    df_true.append((bn, int(event)))
    df_true = pd.DataFrame(df_true, columns = ['base_name', 'frame_number'])
    #%%
#    ANNOTATIONS_FILE = dname / 'collect/single_user_labels/egg_events_corrected.csv'
#    df_true = pd.read_csv(ANNOTATIONS_FILE)
#    df_true = df_true[df_true['set_type'] == 'test']
#    #df_true = df_true[df_true['set_type'] == 'train']
    
    
    df_true = df_true.sort_values(by = 'base_name')
    
    for model_name in models2check:
        
        #src_dir = root_dir / model_name
        
        src_dir = (root_dir / 'predictions' / model_name)
        if not src_dir.is_dir():
            src_dir = (root_dir / 'predictions_bkp' / model_name)
           
        
        #if (src_dir / 'test').is_dir():
        #    src_dir = src_dir / 'test'
        
        errors_dir = src_dir / 'errors'
        
        
        #src_files = list(Path(src_dir).glob('*.p'))
        
        metrics = np.zeros((len(th2check), 3), dtype = np.int)
        
        for bn, bn_data in tqdm(df_true.groupby('base_name'), desc = model_name):
            true_eggs = np.sort(bn_data['frame_number'].values)
            
            src_file = src_dir / (bn + '_eggs.p')
            if not src_file.exists():
                continue
            
            with open(src_file, 'rb') as fid:
                results = pickle.load(fid)
            
            
            predictions = results['predicted_egg_flags']
            
            if predictions.ndim == 2:
                predictions = np.mean(predictions, axis=0)
            
            valid_with_offset = np.concatenate((true_eggs - 1, true_eggs, true_eggs+1))
            valid_with_offset = set(valid_with_offset)
            
            #%%
            
            for ith, th in enumerate(th2check):
                pred_on, = np.where(predictions > th)
                #pred_on, props = find_peaks(predictions, height = th)
                
                wrong_events = set(pred_on) - valid_with_offset
                
                pred_with_offset = np.concatenate((pred_on - 1, pred_on, pred_on+1))
                valid_events = set(true_eggs) & set(pred_with_offset)
                missing_events = set(true_eggs) - valid_events
                
                
                valid_events = list(valid_events)
                wrong_events = list(wrong_events)
                missed_events = list(missing_events)
                
                
                TP = len(valid_events)
                FP = len(wrong_events)
                FN = len(missed_events)
                metrics[ith] += (TP, FP, FN)
                
                
                
                if (th2plot is not None) and ith == ith2plot: #simple comparison did not work due to overflow precision...
                    (xl, xr), (yl, yr) = valid_size
                    ts_file = ts_files[bn]
                    #%%
                    with pd.HDFStore(ts_file) as fid:
                        imgs = fid.get_node('/mask')
                        #print(imgs.shape)
                        
                        for event_type, events2check in [('missed', missed_events), ('wrong', wrong_events)]:
                            dirs2save = errors_dir / event_type
                            dirs2save.mkdir(exist_ok = True, parents = True)
                            
                            for w in sorted(events2check):
                                imgs2check = imgs[w - 2: w + 3, xl:xr, yl:yr]
                                
                                fig, axs = plt.subplots(1, len(imgs2check), 
                                             figsize = (8*len(imgs2check), 8), 
                                             sharex = True, 
                                             sharey = True,
                                             
                                             )
                                for im, ax in zip(imgs2check, axs):
                                    ax.imshow(im, cmap = 'gray')
                                    ax.axis('off')
                                
                                save_name = dirs2save / f'{bn}_T{w}.jpg'
                                plt.suptitle(w, fontsize = 48)
                                fig.savefig(save_name, bbox_inches='tight')
                                plt.close()
                                
                                
        if (metrics != 0).any():
            all_metrics[model_name] = metrics
    
    #%%
    
    scores = {}
    for model_name, metrics in all_metrics.items():
        TP, FP, FN = metrics.T
        
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        scores[model_name] = np.array((P, R, F1)).T
        #%%
        
        
    #%%
    fig, axs = plt.subplots(1,2, figsize = (15, 5))
    
    
    mAPs = []
    for bn, ss in scores.items():
        P, R, F1 = ss.T
        
        
        #calculate the mAP as the area over teh recall vs precision curve
        
        #sort arguments and add paddings
        R_sorted, P_sorted = zip(*sorted(list(zip(R, P))))
        R_sorted = [0.] + list(R_sorted) + [1.]
        P_sorted = [0.] + list(P_sorted) + [0.] #zero at the begining set it to whatever is the value at the lowest recall
        P_smoothed = []
        
        #smooth the precision curve using max(P[i:]) for a given R[i]
        for ii in range(len(P_sorted)):
            P_smoothed.append(max(P_sorted[ii:]))
            
        #numerically integrate the area over the curve (AOC)
        Rs = np.array(R_sorted)
        Ps = np.array(P_smoothed)
        mAP = ((Rs[1:] - Rs[:-1])*Ps[1:]).sum()
        
        th_ind = np.nanargmax(F1)
        
        th = th2check[th_ind] if th_ind == th_ind else np.nan
        
        mAPs.append((mAP, np.nanmax(F1),  th, bn))
       
        
        axs[0].plot(Rs, Ps, '.-',  label = bn)
    
    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
    axs[0].set_ylabel('Precision')
    axs[0].set_xlabel('Recall')
    
    axs[0].legend()
    
    for mAP, F1, th, bn in sorted(mAPs):
        print(mAP, F1, th, bn)
    
    
    for bn, ss in scores.items():
        axs[1].plot(th2check, ss[..., -1], '.-', label = bn)
    
    #plt.plot(sb[..., 0].T, sb[..., 1].T, label = 'other')
    axs[1].set_xlabel('Thresholds')
    axs[1].set_ylabel('F1-score')
    #plt.legend()
    
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    
    
    
    
   