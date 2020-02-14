#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:23:47 2018

@author: ajaver
"""

from tierpsy.features.tierpsy_features import timeseries_feats_columns

import pandas as pd
from tqdm import tqdm
import numpy as np
import numba
from scipy.stats import entropy
from itertools import combinations
import warnings

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.special import comb
import pickle

from pathlib import Path


all_feats = timeseries_feats_columns
all_feats = [x for x in all_feats if not 'path_curvature' in x]


def read_bins_ranges(bin_ranges_file, num_ybins = 31, xbins_args = (-10, 10, 1)):
    
    bin_lims = pd.read_csv(bin_ranges_file, index_col=0)
    xbins = np.arange(*xbins_args)
    num_xbins = len(xbins) + 1
    
    ybins = {}
    for feat in all_feats:
        bot, top = bin_lims.loc[feat].values
        ybins[feat] = np.linspace(bot, top, num_ybins-1)
    
    return xbins, num_xbins, ybins, num_ybins

def digitize_features(centered_data, ybins):
    digitized_data = {}
    for feat in ybins.keys():
        bot, top = ybins[feat][0], ybins[feat][-1]
        dat = np.clip(centered_data[feat], bot + 1e-6, top - 1e-6)
        counts = np.digitize(dat, ybins[feat])
        
        #flag bad rows with -1
        counts[np.isnan(dat)] = -1
        digitized_data[feat] = counts
    
    return digitized_data

@numba.jit(nopython=True)
def calc_histogram_2d(x_digit, y_digit, n_x, n_y):
    H = np.zeros((n_y, n_x), np.int32)
    for ii in range(y_digit.size):
        x = x_digit[ii]
        y = y_digit[ii]
        if y >= 0:
            H[y, x] += 1
    return H

if __name__ == '__main__':
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    #events_dir = Path.home() / 'workspace/WormData/egg_laying_test/' / bn
    events_dir = Path.home() / 'OneDrive - Nexus365/worms/eggs/egg_laying' / bn
    hist_dir = events_dir / 'histograms_data'
    
    files_data = pd.read_pickle(events_dir / 'files_data.pkl')
    strain_basenames = {s : x for s,x in files_data.groupby('strain')}
    
    
    fps = 25
    
    xbins_args = (-10, 10, 1)
    num_ybins = 26
    xbins, num_xbins, ybins, num_ybins = read_bins_ranges(hist_dir / 'bin_limits.csv', num_ybins = num_ybins, xbins_args = xbins_args)
    
    ybins = {k:np.arange(-3, 3+1, 0.2) for k in ybins.keys()}
    num_ybins = ybins[all_feats[0]].size + 1
    
    DIVERGENT_SET = ['CB4856', 'N2',  'DL238', 'CX11314', 'MY23', 'JU775', 'JT11398',
       'EG4725', 'LKC34', 'ED3017', 'MY16', 'JU258']
    
    feats2plot = ['speed', 'curvature_head', 'curvature_tail', 'length', 'relative_to_head_base_radial_velocity_head_tip', 'relative_to_head_base_angular_velocity_head_tip']
    
    
    plots_handles = {}
    for feat in feats2plot:
        fig, axs = plt.subplots(1, len(DIVERGENT_SET), figsize = (20, 5), sharex = True, sharey = True)
        for ax, strain in zip(axs, DIVERGENT_SET):
            ax.set_title(strain)
        fig.suptitle(feat)
        
        plots_handles[feat] = fig, axs
        
        
    for istrain, strain in enumerate(DIVERGENT_SET):
        basenames = strain_basenames[strain]['basename'].values
        
        strain_ts_data = []
        for bn in tqdm(basenames, desc = strain):
            ts_file = events_dir / f'{bn}_timeseries.pkl'
            if ts_file.exists():
                df_ts = pd.read_pickle(ts_file)
                #strain_ts_data.append(df_ts)
                
                
                m = df_ts.mean()
                s = df_ts.std()
                df_ts_norm = (df_ts - m) / s
                
                cols = ['worm_index', 'timestamp' , 'video_id', 'timestamp_centered']
                df_ts_norm[cols] = df_ts[cols]
                strain_ts_data.append(df_ts_norm)
                
                
        strain_ts_data = pd.concat(strain_ts_data)
        
        
        df = strain_ts_data
        df['time_centered'] = df['timestamp_centered']/fps
        valid = (df['time_centered']>=xbins[0]) & (df['time_centered']<=xbins[-1])
        df = df[valid]
        
        x_digit = np.digitize(df['time_centered'], xbins)
        digitized_feats = digitize_features(df, ybins)
        
        min_valid_counts = 500
        histograms = {}
        
        
        
        
        for feat in feats2plot:
            y_digit = digitized_feats[feat]
            assert y_digit.size == x_digit.size
            
            counts = calc_histogram_2d(x_digit, y_digit, num_xbins, num_ybins)
            
            N = counts.sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                H = counts/N
            H[:, N<min_valid_counts] = np.nan
            
            histograms[feat] = (counts, N, H)
            
            fig, axs = plots_handles[feat]
            ax = axs[istrain]
            ax.imshow(H)
            
        #fig.suptitle(strain)
    #%%
    
#    save_dir = './data'
#    
#    import multiprocessing as mp
#    def _process(strain):
#        plot_input = calculate_strain_JSD(strain, save_dir = save_dir, delT = 5)
#        save_plots(*plot_input)
#        
#    exp_df = pd.read_csv(os.path.join(save_dir, 'index.csv'), index_col=0)
#    
#    uStrains = exp_df['strain'].unique()
#    
#    
#    #be careful with memory here...
#    p = mp.Pool(len(uStrains))
#    p.map(_process, uStrains)
    
    
    #for strain in tqdm.tqdm(exp_df['strain'].unique()):
        
         

