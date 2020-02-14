#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:33:10 2019

@author: avelinojaver
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import datetime
import numpy as np


from tensorboardX import SummaryWriter
from cell_localization.utils import save_checkpoint, get_optimizer,get_scheduler,  get_device

from pathlib import Path

from flow import SnippetsFullFlow, SnippetsRandomFlow
from models import get_model




def get_warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    #https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(basename, model, criterion, optimizer, warmup_lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    train_avg_loss = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, target in pbar:
        
        images = images.to(device)
        target = target.to(device)
        
        prediction = model(images)
        loss = criterion(prediction, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if warmup_lr_scheduler is not None:
            warmup_lr_scheduler.step()
        
        train_avg_loss += loss.item()
     
        
    train_avg_loss = train_avg_loss/len(data_loader)
    
    #save data into the logger
    logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
    
    return train_avg_loss

@torch.no_grad()
def evaluate_one_epoch(basename, model, criterion, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.eval()
    header = f'{basename} Test Epoch: [{epoch}]'
    
    test_avg_loss = 0
    
    metrics = np.zeros(4)
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, target in pbar:
        
        images = images.to(device)
        target = target.to(device)
        
        prediction = model(images)
        
        loss = criterion(prediction, target)
        test_avg_loss += loss.item()
        
        
        th_true = 0.6
        
        prediction = prediction.squeeze(-1)
        #prediction = torch.sigmoid(prediction) # TODO once you figure out what model to use...
        
        target_true = (target >= th_true)
        is_correct = target_true == (prediction >= th_true)
        
        true_pos = is_correct[target_true].sum().item()
        true_neg = is_correct[~target_true].sum().item()
        
        total_pos = target.sum().item()
        total_neg = target.numel() - total_pos
        
        metrics += (true_pos, total_pos, true_neg, total_neg)
    
    true_pos, total_pos, true_neg, total_neg = metrics
    P_pos = true_pos/total_pos
    P_neg = true_neg/total_neg
    
    test_avg_loss = test_avg_loss/len(data_loader)
    
    #save data into the logger
    logger.add_scalar('val_epoch_loss', test_avg_loss, epoch)
    logger.add_scalar('P_pos', P_pos, epoch)
    logger.add_scalar('P_neg', P_neg, epoch)
    
    return test_avg_loss


def from_pretrained(model, pretrained_path):
    print(f"Loading Pretrained weights from {pretrained_path}")
    state = torch.load(pretrained_path, map_location = 'cpu')
    state_dict = model.state_dict()
    pretrained_dict = state['state_dict']
    
    for k in state_dict:
        if k in pretrained_dict:
            state_dict[k] = pretrained_dict[k]   
    model.load_state_dict(state_dict)
    
    return model

def main(
        data_type = 'v3',#'v1-0.5x',
        model_name = '',
        loss_name = 'BCE', #MSEp2
        cuda_id = 0,
         batch_size = 4,
         n_epochs = 200,
         samples_per_epoch = 2500,
         num_workers = 1,
         save_frequency = 200,
         lr = 1e-5,
         weight_decay = 0.0,
         optimizer_name = 'adam',
         lr_scheduler_name = '',
         log_dir = None,
         warmup_epochs = 0,
         warmup_factor = 1/1000,
         resume_path = None
         ):
    
    root_dir = Path.home() / 'workspace/WormData/egg_laying/single_worm'
    
    if log_dir is None:
        log_dir = root_dir / 'results'
        
        
    data_d = data_type
    flow_train_argkws = {}
        
    if 'mixed-setups' in data_type:
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
            
    
    elif data_type.endswith('+hard'):
        data_d = data_type[:-5]
        flow_train_argkws = dict(
                max_offset_per_frame = 50,
                 motion_blur_range = (5, 45),
                 zoom_range = (0.9, 1.1),
                 int_aug_offset = (-0.2, 0.2), 
                 int_aug_expansion = (0.75, 1.2), 
                 convolve_egg_flag_kernel = [0, 0.15, 1., 0.3, 0.15]
                )
    
    
    if data_d == 'v1-0.5x':
        data_dir = root_dir / 'data/v1_0.5x/'
    else:
        data_dir = root_dir / 'data' / data_d
    
    if not data_dir.exists():
        raise ValueError(f'Not Implemented `{data_type}`. Directory `{data_dir}` does not exists.')
    
    mm = model_name.partition('+')[0]
    #model = get_model(mm, n_in = 1, n_out = 2)
    model = get_model(mm, n_in = 1, n_out = 1)
    
    
    if 'pretrained' in model_name:
        bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
        pretrained_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam-masks/' / bn / 'model_best.pth.tar'
        model = from_pretrained(model, pretrained_path)
    
    if 'frozen' in model_name:
        print('`mapping_network` frozen')
        for p in model.mapping_network.parameters():
               p.requires_grad = False
    
    if resume_path is not None:
        state = torch.load(resume_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model_name += '+R'
    
    
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    train_flow = SnippetsRandomFlow(train_dir, samples_per_epoch = samples_per_epoch, **flow_train_argkws)
    val_flow = SnippetsFullFlow(test_dir)
    
    if loss_name == 'BCE':
        _criterion = nn.BCEWithLogitsLoss()
        def criterion(prediction, target):
            return _criterion(prediction.squeeze(-1), target)
    elif loss_name == 'BCEp2':
        _criterion = nn.BCEWithLogitsLoss()
        pad = 2
        def criterion(prediction, target):
            return _criterion(prediction[:, pad:-pad].squeeze(-1), target[:, pad:-pad])
    elif loss_name == 'MSEp2':
        pad = 2
        _criterion = nn.MSELoss()
        def criterion(prediction, target):
            return _criterion(prediction[:, pad:-pad].squeeze(-1), target[:, pad:-pad])
    
    
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay, weigth_decay_no_bias = True)
    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer)
    
    device = get_device(cuda_id)
    
    
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers
                            )
    val_loader = DataLoader(val_flow, 
                            batch_size = 1, 
                            shuffle=True, 
                            num_workers=num_workers
                            )
    
    model = model.to(device)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_prefix = f'WT+{data_type}_{model_name}_{loss_name}_{date_str}_{optimizer_name}-{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    if warmup_epochs > 0:
        save_prefix += f'_warmup{warmup_epochs}'
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e8
    pbar_epoch = tqdm.trange(n_epochs)
    
    
    if warmup_epochs > 0:
        warmup_iters = len(train_loader) * warmup_epochs - 1
        warmup_lr_scheduler = get_warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    for epoch in pbar_epoch:
        if epoch >= warmup_epochs:
            warmup_lr_scheduler = None
        
        train_one_epoch(save_prefix, 
                         model, 
                         criterion,
                         optimizer, 
                         warmup_lr_scheduler, 
                         train_loader, 
                         device, 
                         epoch, 
                         logger
                         )
        
        if lr_scheduler is not None and epoch >= warmup_epochs:
            lr_scheduler.step()
        
        val_loss = evaluate_one_epoch(save_prefix, 
                           model, 
                           criterion,
                           val_loader, 
                           device, 
                           epoch, 
                           logger
                           )
        
        
        
        desc = f'epoch {epoch} , val_loss={val_loss}'
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 
if __name__ == '__main__':
    import fire
    fire.Fire(main)
        