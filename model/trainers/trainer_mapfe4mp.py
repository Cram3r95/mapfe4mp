#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Trainer

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import gc
import os
import numpy as np
import pdb
import time
import sys
import copy

# DL & Math imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.tensorboard import SummaryWriter

# Custom imports

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
from model.models.mapfe4mp import TrajectoryGenerator
from model.modules.losses import l2_loss_multimodal, mse, pytorch_neg_multi_log_likelihood_batch, evaluate_feasible_area_prediction
from model.modules.evaluation_metrics import displacement_error, final_displacement_error
from model.datasets.argoverse.dataset_utils import relative_to_abs_multimodal
from model.datasets.argoverse.map_functions import MapFeaturesUtils
from model.utils.checkpoint_data import Checkpoint, get_total_norm
from model.utils.utils import create_weights

#######################################

# Global variables

map_features_utils_instance = MapFeaturesUtils()

torch.backends.cudnn.benchmark = True
scaler = GradScaler()
current_cuda = None
absolute_root_folder = None

USE_SCALER = False # Do not use this scaler, first learn how to use it properly
CHECK_ACCURACY_TRAIN = False
CHECK_ACCURACY_VAL = True
MAX_TIME_TO_CHECK_TRAIN = 60 # minutes
MAX_TIME_TO_CHECK_VAL = 60 # minutes
MAX_TIME_PATIENCE_LR_SCHEDULER = 60 # 60 # minutes

min_ade_ = 50000
g_lr = 0.001

# Aux functions

def get_best_predictions(pred, best_pred_indeces):
    """
    pred: batch_size x num_modes x pred_len x data_dim
    best_pred_indeces: batch_size x 1

    Take the best prediction (best mode) according to the best confidence for each sequence
    """
    return pred[torch.arange(pred.shape[0]), best_pred_indeces, :, :].squeeze()

def get_lr(optimizer):
    """
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_weights(module):
    """
    """
    classname = module.__class__.__name__
    try:
        if classname.find('Linear') != -1 and classname != "LinearRes":
            nn.init.kaiming_normal_(module.weight)
    except: # Custom layers
        keys = module.__dict__['_modules'].keys()

        for key in keys:
            if key.capitalize().find('Linear') != -1:
                nn.init.kaiming_normal_(module.__dict__['_modules'][key].weight)

def get_dtypes(use_gpu):
    """
    """
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

# Aux functions losses

def calculate_mse_gt_loss_multimodal(gt, pred, loss_f, compute_ade=True, compute_fde=True, w_loss=None):
    """
    gt: (pred_len, batch_size, data_dim)
    pred: (batch_size, num_modes, pred_len, data_dim)
    """
    
    batch_size, num_modes, pred_len, data_dim = pred.shape

    pred = pred.permute(1,2,0,3) # num_modes, pred_len, batch_size, data_dim

    loss_ade = torch.zeros(1).to(pred)
    loss_fde = torch.zeros(1).to(pred)

    for i in range(num_modes):
        if compute_ade: loss_ade += loss_f(pred[i,:,:,:], gt, w_loss)
        if compute_fde: loss_fde += loss_f(pred[i][-1].unsqueeze(0), gt[-1].unsqueeze(0), w_loss)

    return loss_ade/num_modes, loss_fde/num_modes

def calculate_mse_gt_loss_unimodal(gt, pred, loss_f, w_loss=None):
    """
    gt: (pred_len, batch_size, data_dim)
    pred: (batch_size, pred_len, data_dim) (only the predictions with the best confidence for each sequence)
    """

    pred = pred.permute(1,0,2) # pred_len, batch_size, data_dim

    loss_ade = loss_f(pred, gt, w_loss).to(pred)
    loss_fde = loss_f(pred[-1].unsqueeze(0), gt[-1].unsqueeze(0), w_loss).to(pred)

    return loss_ade, loss_fde

def calculate_mse_centerlines_loss(centerlines, pred, loss_f, w_loss=None):
    """
    centerlines: (num_modes, batch_size, centerline_length, data_dim)
    pred: (batch_size, num_modes, pred_len, data_dim)

    N.B. The centerline length and pred_len must match. Otherwise, use the function
    calculate_mse_centerlines_loss_interpolating to get first the closest centerline
    waypoint to the first prediction and then interpolate pred_len points to the goal
    """

    batch_size, num_modes, pred_len, data_dim = pred.shape

    pred = pred.permute(1,2,0,3)
    centerlines = centerlines.permute(0,2,1,3)

    loss_ade = torch.zeros(1).to(pred)
    loss_fde = torch.zeros(1).to(pred)

    for i in range(num_modes):
        loss_ade += loss_f(pred[i,:,:,:], centerlines[i,:,:,:], w_loss)
        loss_fde += loss_f(pred[i][-1].unsqueeze(0), centerlines[i][-1].unsqueeze(0), w_loss)

    return loss_ade/num_modes, loss_fde/num_modes

def calculate_mse_centerlines_loss_interpolating(relevant_centerlines, pred, loss_f, w_loss=None):
    """
    relevant_centerlines: (num_modes, batch_size, centerline_length, data_dim)
    pred: (batch_size, num_modes, pred_len, data_dim)
    """
    
    batch_size, num_modes, pred_len, data_dim = pred.shape

    relevant_centerlines_ = relevant_centerlines.clone()
    centerline_length = relevant_centerlines_.shape[2]
    
    pred = pred.permute(1,2,0,3)
    loss_ade = torch.zeros(1).to(pred)
    loss_fde = torch.zeros(1).to(pred)

    for mode in range(num_modes):
        mode_centerlines = relevant_centerlines_[mode,:,:,:] # batch_size x centerline_length x 2
        mode_l2 = torch.norm(mode_centerlines,dim=2) # get distances to origin (abs coordinates)
        index_min = torch.argmin(mode_l2,dim=1)

        c1 = torch.where(index_min <= (centerline_length-pred_len))[0] # Take pred_len points ahead from these centerlines
        c2 = torch.where(index_min > (centerline_length-pred_len))[0] # From this point, interpolate pred_len points to the end of the centerline 
        
        filtered_relevant_centerlines = []
        for i in range(batch_size):
            assert (i in c1) or (i in c2)
            if i in c1:
                centerline = mode_centerlines[i,index_min[i]:index_min[i]+pred_len,:]
                filtered_relevant_centerlines.append(centerline)
            elif i in c2: # interpolate
                centerline = mode_centerlines[i,index_min[i]:,:] # Take to the end
                interpolated_centerline = map_features_utils_instance.interpolate_centerline(centerline,max_points=pred_len)
                try:
                    assert interpolated_centerline.shape[0] == pred_len 
                except:
                    pdb.set_trace()
                filtered_relevant_centerlines.append(interpolated_centerline)

        pdb.set_trace()
        # Check if from those indeces to the end of the centerline there are at least pred_len points


        # loss_ade += loss_f(pred[mode,:,:,:], gt, w_loss)
        # loss_fde += loss_f(pred[mode][-1].unsqueeze(0), gt[-1].unsqueeze(0), w_loss)

    return loss_ade/num_modes, loss_fde/num_modes

def calculate_nll_loss(gt, pred, loss_f, confidences):
    """
    NLL = Negative Log-Likelihood
    Compute NLL w.r.t. the groundtruth
    """
    time, bs, _ = gt.shape
    gt = gt.permute(1,0,2)
    avails = torch.ones(bs,time).cuda(current_cuda)
    loss = loss_f(
        gt, 
        pred,
        confidences,
        avails
    )
    return loss

def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    """
    """
    g_l2_loss_abs = l2_loss_multimodal(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss_multimodal(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')

    return g_l2_loss_abs, g_l2_loss_rel

# TODO: Implement linear and non_linear for cal_ade and cal_fde

def cal_ade_multimodal(pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped):
    b,m,t,_ = pred_traj_fake.shape
    ade = []
    for i in range(b):
        _ade = []
        for j in range(m):
            __ade = displacement_error(
                pred_traj_fake[i,j,:,:].unsqueeze(0).permute(1,0,2), pred_traj_gt[:,i,:].unsqueeze(1), consider_ped
            )
            _ade.append(__ade.item())
        ade.append(_ade)
    ade = np.array(ade)
    min_ade = np.min(ade, 1)
    return ade, min_ade

def cal_fde_multimodal(
    pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped
):
    b,m,t,_ = pred_traj_fake.shape
    fde = []
    for i in range(b):
        _fde = []
        for j in range(m):
            __fde = final_displacement_error(
                pred_traj_fake[i,j,-1,:].unsqueeze(0), pred_traj_gt[-1,i].unsqueeze(0), consider_ped
            )
            _fde.append(__fde.item())
        fde.append(_fde)
    fde = np.array(fde)
    min_fde = np.min(fde, 1)
    return fde, min_fde

# Main function

def model_trainer(config, logger):
    """
    """

    # Aux variables

    hyperparameters = config.hyperparameters
    optim_parameters = config.optim_parameters

    ## Set specific device for both data and model

    global current_cuda
    current_cuda = torch.device(f"cuda:{config.device_gpu}")
    device = torch.device(current_cuda if torch.cuda.is_available() else "cpu")
    
    long_dtype, float_dtype = get_dtypes(config.use_gpu)

    logger.info('Configuration: ')
    logger.info(config)

    global absolute_root_folder
    absolute_root_folder = os.path.join(config.base_dir,config.dataset.path)

    # Initialize train dataloader (N.B. If applied, data augmentation is only used during training!)

    logger.info("Initializing train dataset") 

    data_train = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                   root_folder=config.dataset.path,
                                                   imgs_folder=config.dataset.imgs_folder,
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split="train",
                                                   split_percentage=config.dataset.split_percentage,
                                                   batch_size=config.dataset.batch_size,
                                                   class_balance=config.dataset.class_balance,
                                                   obs_origin=config.hyperparameters.obs_origin,
                                                   data_augmentation=config.dataset.data_augmentation,
                                                   physical_context=config.hyperparameters.physical_context,
                                                   extra_data_train=config.dataset.extra_data_train,
                                                   preprocess_data=config.dataset.preprocess_data,
                                                   save_data=config.dataset.save_data)

    train_loader = DataLoader(data_train,
                              batch_size=config.dataset.batch_size,
                              shuffle=config.dataset.shuffle,
                              num_workers=config.dataset.num_workers,
                              collate_fn=seq_collate)

    # Initialize validation dataloader

    logger.info("Initializing val dataset")
    data_val = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                 root_folder=config.dataset.path,
                                                 imgs_folder=config.dataset.imgs_folder,
                                                 obs_len=config.hyperparameters.obs_len,
                                                 pred_len=config.hyperparameters.pred_len,
                                                 distance_threshold=config.hyperparameters.distance_threshold,
                                                 split="val",
                                                 split_percentage=config.dataset.split_percentage,
                                                 class_balance=-1,
                                                 obs_origin=config.hyperparameters.obs_origin,
                                                 physical_context=config.hyperparameters.physical_context,
                                                 extra_data_train=config.dataset.extra_data_train,
                                                 preprocess_data=config.dataset.preprocess_data,
                                                 save_data=config.dataset.save_data)
                                     
    val_loader = DataLoader(data_val,
                            batch_size=config.dataset.batch_size,
                            shuffle=False,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)

    # Initialize motion prediction generator and optimizer

    try:
        generator = TrajectoryGenerator(PHYSICAL_CONTEXT=hyperparameters.physical_context)
    except Exception as e:
        print("Exception: ", e)
        generator = TrajectoryGenerator()
        
    generator.to(device)
    generator.apply(init_weights)
    generator.type(float_dtype).train() # train mode (if you compute metrics -> .eval() mode)

    optimizer_g = optim.Adam(generator.parameters(), lr=optim_parameters.g_learning_rate, weight_decay=optim_parameters.g_weight_decay)

    ## Restore from checkpoint if specified in config file

    restore_path = None
    if hyperparameters.checkpoint_start_from is not None:
        restore_path = hyperparameters.checkpoint_start_from

    previous_t = 0 # Previous iterations

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path, map_location=current_cuda)
        
        generator.load_state_dict(checkpoint.config_cp['g_best_state'], strict=False)
        optimizer_g.load_state_dict(checkpoint.config_cp['g_optim_state'])

        t = checkpoint.config_cp['counters']['t'] # Restore num_iters and epochs from checkpoint
        epoch = checkpoint.config_cp['counters']['epoch']
        previous_t = copy.copy(t)
        # t,epoch = 0,0 # set to 0, though it starts from a specific checkpoint. 
                        # Repeat all iterations/epochs again
                        
        checkpoint.config_cp['restore_ts'].append(t)
    else:
        # Start from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = Checkpoint()

    logger.info('Generator model:')
    logger.info(generator)
    logger.info('Optimizer:')
    logger.info(optimizer_g)

    ## Compute total number of iterations (iterations_per_epoch * num_epochs)
    
    iterations_per_epoch = len(data_train) / config.dataset.batch_size
    
    if hyperparameters.num_epochs:
        hyperparameters.num_iterations = int(iterations_per_epoch * hyperparameters.num_epochs) # compute total iterations
        assert hyperparameters.num_iterations != 0

    ## Compute num iterations to calculate accuracy (both train and val)

    hyperparameters.checkpoint_train_every = round(hyperparameters.checkpoint_train_percentage * hyperparameters.num_iterations)

    hyperparameters.checkpoint_val_every = round(hyperparameters.checkpoint_val_percentage * hyperparameters.num_iterations)
    if hyperparameters.checkpoint_val_every < hyperparameters.print_every:
        hyperparameters.checkpoint_val_every = hyperparameters.print_every

    # Scheduler and loss functions
    
    if hyperparameters.lr_scheduler: 
        lr_scheduler_patience = int(hyperparameters.lr_epoch_percentage_patience * hyperparameters.num_iterations) # Every N epochs
        
        if lr_scheduler_patience < hyperparameters.print_every:
            lr_scheduler_patience = hyperparameters.print_every * 2.1
        
        scheduler_g = lrs.ReduceLROnPlateau(optimizer_g, "min", min_lr=hyperparameters.lr_min, 
                                            verbose=True, factor=hyperparameters.lr_decay_factor, 
                                            patience=lr_scheduler_patience)

    if hyperparameters.loss_type_g == "mse" or hyperparameters.loss_type_g == "mse_w":
        loss_f = mse
    elif hyperparameters.loss_type_g == "nll":
        loss_f = pytorch_neg_multi_log_likelihood_batch
    elif hyperparameters.loss_type_g == "mse+fa" or hyperparameters.loss_type_g == "mse_w+fa":
        loss_f = {
            "mse": mse,
            "fa": evaluate_feasible_area_prediction
        }
    elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":
        loss_f = {
            "mse": mse,
            "nll": pytorch_neg_multi_log_likelihood_batch
        }
    elif hyperparameters.loss_type_g == "centerlines+gt":
        loss_f = {
            "mse": mse,
            "nll": pytorch_neg_multi_log_likelihood_batch
        }
    else:
        assert 1 == 0, "loss_type_g is not correct"

    min_weight = 1
    max_weight = 4
    w_loss = create_weights(config.dataset.batch_size, 
                            min_weight, 
                            max_weight).cuda(current_cuda) # batch_size x pred_len, from min_weight to max_weight

    # Tensorboard

    if hyperparameters.tensorboard_active:
        exp_path = os.path.join(
            config.base_dir, hyperparameters.output_dir, "tensorboard_logs"
        )
        os.makedirs(exp_path, exist_ok=True)
        writer = SummaryWriter(exp_path)

    ###################################
    #### Main Loop. Start training ####
    ###################################

    # N.B. Despite the fact the main loop depends on t, since we consider all previous iterations
    # we are only training right now the specified number of epochs / iterations (given by current_iteration and
    # hyperparameters.num_iterations)
    
    num_analyzed_seqs = 0
    time_iterations = float(0)
    time_per_iteration = float(0)
    time_seq_collate = float(0)
    time_per_seq_collate = float(0)
    current_iteration = 0 # Current iteration, regardless the model had previous training
    flag_check_every = False

    # while t < hyperparameters.num_iterations + previous_t:
    global g_lr
    while g_lr > hyperparameters.lr_min:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))

        start_seq_collate = time.time()
        for batch in train_loader:
            end_seq_collate = time.time()

            start = time.time()
            losses_g = generator_step(hyperparameters, batch, generator, optimizer_g, loss_f, w_loss)
            end = time.time()
            checkpoint.config_cp["norm_g"].append(get_total_norm(generator.parameters()))
            
            time_iterations += end - start
            time_per_iteration = time_iterations / (current_iteration+1)

            time_seq_collate += end_seq_collate - start_seq_collate
            time_per_iteration = time_seq_collate / (current_iteration+1)

            # Compute approximate time to compute train and val metrics

            if not flag_check_every:
                print("time per iteration: ", time_per_iteration)
                print("time per seq collate: ", time_per_seq_collate)
                total_time = time_per_seq_collate + time_per_iteration

                seconds_to_check_train = total_time*hyperparameters.checkpoint_train_every
                seconds_to_check_val = total_time*hyperparameters.checkpoint_val_every
                seconds_lr_patience = total_time*lr_scheduler_patience

                if seconds_to_check_train > MAX_TIME_TO_CHECK_TRAIN*60:
                    hyperparameters.checkpoint_train_every = round((MAX_TIME_TO_CHECK_TRAIN*60) / total_time)
                if seconds_to_check_val > MAX_TIME_TO_CHECK_VAL*60:
                    hyperparameters.checkpoint_val_every = round((MAX_TIME_TO_CHECK_VAL*60) / total_time)
                if seconds_lr_patience > MAX_TIME_PATIENCE_LR_SCHEDULER*60:
                    lr_scheduler_patience = round((MAX_TIME_PATIENCE_LR_SCHEDULER*60) / total_time)

                    scheduler_g = lrs.ReduceLROnPlateau(optimizer_g, "min", min_lr=hyperparameters.lr_min, 
                                                        verbose=True, factor=hyperparameters.lr_decay_factor, 
                                                        patience=lr_scheduler_patience)

                mins_to_check_train = round((total_time*hyperparameters.checkpoint_train_every)/60)
                mins_to_check_val = round((total_time*hyperparameters.checkpoint_val_every)/60)
                mins_to_complete = round((total_time*hyperparameters.num_iterations)/60)
                mins_to_decrease_lr = round((total_time*lr_scheduler_patience)/60)

                logger.info(
                            '\n Iterations per epoch: {} \n \
                                Total epochs: {} \n \
                                Total iterations: {} (Aprox. {} min) \n \
                                Checkpoint train every: {} iterations (Aprox. {} min) \n \
                                Checkpoint val every: {} iterations (Aprox. {} min) \n \
                                Decrease LR if condition not met every: {} (Aprox. {} min)'
                                .format(iterations_per_epoch, 
                                        hyperparameters.num_epochs,
                                        hyperparameters.num_iterations,
                                        mins_to_complete,
                                        hyperparameters.checkpoint_train_every,
                                        mins_to_check_train,
                                        hyperparameters.checkpoint_val_every,
                                        mins_to_check_val,
                                        lr_scheduler_patience,
                                        mins_to_decrease_lr))

                flag_check_every = True

            # N.B. t represents the num of iteration. However, it is difficult to compare different plots in
            # tensorboard if every iteration has a different batch size. So, we multiply the current iteration
            # multiplied by the batch size in order to have in the X-axis the num of processed sequences 

            num_analyzed_seqs = (t+1) * config.dataset.batch_size

            # Print losses info
            
            if current_iteration % hyperparameters.print_every == 0:
                logger.info('Iteration = {} / {}'.format(t+1, hyperparameters.num_iterations + previous_t))
                logger.info('Time per iteration: {}'.format(time_per_iteration))
                logger.info('Time per seq collate: {}'.format(time_per_seq_collate))
                logger.info('Num of analyzed sequences: {}'.format(num_analyzed_seqs))

                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, num_analyzed_seqs)
                    if k not in checkpoint.config_cp["G_losses"].keys():
                        checkpoint.config_cp["G_losses"][k] = [] 
                    checkpoint.config_cp["G_losses"][k].append(v)
                checkpoint.config_cp["losses_ts"].append(t)

            # Check training metrics

            if (CHECK_ACCURACY_TRAIN and current_iteration > 0 
                and current_iteration % hyperparameters.checkpoint_train_every == 0):

                logger.info('Checking stats on train ...')
                split = "train"

                metrics_train = check_accuracy(hyperparameters, train_loader, generator, split=split)

                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, num_analyzed_seqs)
                    if k not in checkpoint.config_cp["metrics_train"].keys():
                        checkpoint.config_cp["metrics_train"][k] = []
                    checkpoint.config_cp["metrics_train"][k].append(v)

            # Check validation metrics

            if (CHECK_ACCURACY_VAL and current_iteration > 0 
                and current_iteration % hyperparameters.checkpoint_val_every == 0):

                logger.info('Checking stats on val ...')
                split = "val"

                # Compute val metrics        

                metrics_val = check_accuracy(
                    hyperparameters, val_loader, generator, split=split
                )

                checkpoint.config_cp["counters"]["t"] = t
                checkpoint.config_cp["counters"]["epoch"] = epoch
                checkpoint.config_cp["sample_ts"].append(t)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, num_analyzed_seqs)
                    if k not in checkpoint.config_cp["metrics_val"].keys():
                        checkpoint.config_cp["metrics_val"][k] = []
                    checkpoint.config_cp["metrics_val"][k].append(v)

                # Get previous best metrics in order to compare

                global min_ade_
                min_ade_ = min(checkpoint.config_cp["metrics_val"][f'{split}_ade'])
                min_fde = min(checkpoint.config_cp["metrics_val"][f'{split}_fde'])
                min_ade_nl = min(checkpoint.config_cp["metrics_val"][f'{split}_ade_nl'])

                logger.info("Min ADE: {}".format(min_ade_))
                logger.info("Min FDE: {}".format(min_fde))

                # Compare with previous validation metrics

                if metrics_val[f'{split}_ade'] <= min_ade_:
                    logger.info('New low for avg_disp_error')
                    checkpoint.config_cp["best_t"] = t
                    checkpoint.config_cp["g_best_state"] = generator.state_dict()

                    # If val_min_ade < min_ade_, save another checkpoint with model weights and
                    # optimizer state

                    checkpoint.config_cp["g_state"] = generator.state_dict()
                    checkpoint.config_cp["g_optim_state"] = optimizer_g.state_dict()
                    checkpoint_path = os.path.join(
                        config.base_dir, hyperparameters.output_dir, "{}_{}_with_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    logger.info('Done.')

                    # Save a checkpoint with no model weights by making a shallow
                    # copy of the checkpoint excluding some items
                    checkpoint_path = os.path.join(
                        config.base_dir, hyperparameters.output_dir, "{}_{}_no_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    key_blacklist = [
                        'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                        'g_optim_state', 'd_optim_state', 'd_best_state',
                        'd_best_nl_state'
                    ]
                    small_checkpoint = {}
                    for k, v in checkpoint.config_cp.items():
                        if k not in key_blacklist:
                            small_checkpoint[k] = v
                    torch.save(small_checkpoint, checkpoint_path)
                    logger.info('Done.')

                if metrics_val[f'{split}_ade_nl'] <= min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint.config_cp["best_t_nl"] = t
                    checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()

            t += 1
            current_iteration += 1
    
            if hyperparameters.lr_scheduler:
                # scheduler_g.step(losses_g["G_total_loss"]) # Do not use training total loss 
                #                                              to reduce learning rate

                scheduler_g.step(min_ade_)
                g_lr = get_lr(optimizer_g)
                writer.add_scalar("G_lr", g_lr, num_analyzed_seqs)

            start_seq_collate = time.time()

    logger.info("Training finished")

    # Check train and validation metrics once the training is finished
    # Because you check every N iterations, but not specifically the last one

    num_analyzed_seqs = t * config.dataset.batch_size

    ## Check train metrics

    logger.info('Checking stats on train ...')
    split = "train"

    metrics_train = check_accuracy(
        hyperparameters, train_loader, generator, split=split
    )

    for k, v in sorted(metrics_train.items()):
        logger.info('  [train] {}: {:.3f}'.format(k, v))
        if hyperparameters.tensorboard_active:
            writer.add_scalar(k, v, num_analyzed_seqs)
        if k not in checkpoint.config_cp["metrics_train"].keys():
            checkpoint.config_cp["metrics_train"][k] = []
        checkpoint.config_cp["metrics_train"][k].append(v)

    ## Check val metrics

    logger.info('Checking stats on val ...')
    split = "val"

    checkpoint.config_cp["counters"]["t"] = t
    checkpoint.config_cp["counters"]["epoch"] = epoch
    checkpoint.config_cp["sample_ts"].append(t)

    metrics_val = check_accuracy(
        hyperparameters, val_loader, generator, split=split
    )

    for k, v in sorted(metrics_val.items()):
        logger.info('  [val] {}: {:.3f}'.format(k, v))
        if hyperparameters.tensorboard_active:
            writer.add_scalar(k, v, num_analyzed_seqs)
        if k not in checkpoint.config_cp["metrics_val"].keys():
            checkpoint.config_cp["metrics_val"][k] = []
        checkpoint.config_cp["metrics_val"][k].append(v)

    min_ade = min(checkpoint.config_cp["metrics_val"][f'{split}_ade'])
    min_fde = min(checkpoint.config_cp["metrics_val"][f'{split}_fde'])
    min_ade_nl = min(checkpoint.config_cp["metrics_val"][f'{split}_ade_nl'])
    logger.info("Min ADE: {}".format(min_ade))
    logger.info("Min FDE: {}".format(min_fde))
    if metrics_val[f'{split}_ade'] <= min_ade:
        logger.info('New low for avg_disp_error')
        checkpoint.config_cp["best_t"] = t
        checkpoint.config_cp["g_best_state"] = generator.state_dict()

    if metrics_val[f'{split}_ade_nl'] <= min_ade_nl:
        logger.info('New low for avg_disp_error_nl')
        checkpoint.config_cp["best_t_nl"] = t
        checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()

    # If val_min_ade, save another checkpoint with model weights and
    # optimizer state

    if metrics_val[f'{split}_ade'] <= min_ade:
        checkpoint.config_cp["g_state"] = generator.state_dict()
        checkpoint.config_cp["g_optim_state"] = optimizer_g.state_dict()
        checkpoint_path = os.path.join(
            config.base_dir, hyperparameters.output_dir, "{}_{}_with_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
        )
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(checkpoint, checkpoint_path)
        logger.info('Done.')

        # Save a checkpoint with no model weights by making a shallow. 
        # Copy of the checkpoint excluding some items

        checkpoint_path = os.path.join(
            config.base_dir, hyperparameters.output_dir, "{}_{}_no_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
        )
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        key_blacklist = [
            'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
            'g_optim_state', 'd_optim_state', 'd_best_state',
            'd_best_nl_state'
        ]
        small_checkpoint = {}
        for k, v in checkpoint.config_cp.items():
            if k not in key_blacklist:
                small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)
        logger.info('Done.')

def generator_step(hyperparameters, batch, generator, optimizer_g, 
                   loss_f, w_loss=None, split="train"):
    """
    """

    # Load data in device

    if hyperparameters.physical_context != "plausible_centerlines+area":
        # Here the physical info is a single tensor

        batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
        loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm, phy_info) = batch
    elif hyperparameters.physical_context == "plausible_centerlines+area":

        # TODO: In order to improve this, seq_collate should return a dictionary instead of a tuple
        # in order to avoid hardcoded positions
        phy_info = batch[-1] # phy_info should be in the last position!

        batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
        loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm) = batch
    
    batch_size = seq_start_end.shape[0]

    # Take (if specified) data of only the AGENT of interest

    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
        loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
    else:
        loss_mask = loss_mask[:, hyperparameters.obs_len:]

    # Forward

    optimizer_g.zero_grad()

    with autocast():
        # Forward (If agent_idx != None, model prediction refers only to target agent)

        relevant_centerlines = []
        if hyperparameters.physical_context == "plausible_centerlines+area":
            # TODO: Encapsulate this as a function

            # Plausible area (discretized)

            plausible_area_array = np.array([curr_phy_info["plausible_area_abs"] for curr_phy_info in phy_info]).astype(np.float32)
            plausible_area = torch.from_numpy(plausible_area_array).type(torch.float).cuda(obs_traj.device)

            # Compute relevant centerlines

            num_relevant_centerlines = [len(curr_phy_info["relevant_centerlines_abs"]) for curr_phy_info in phy_info]
            aux_indeces_list = []
            for num_ in num_relevant_centerlines:
                if num_ >= hyperparameters.num_modes: # Limit to the first hyperparameters.num_modes centerlines. We assume
                                                    # the oracle is here. TODO: Force this from the preprocessing 
                    aux_indeces = np.arange(hyperparameters.num_modes)
                    aux_indeces_list.append(aux_indeces)
                else: # We have less centerlines than hyperparameters.num_modes
                    quotient = int(hyperparameters.num_modes / num_)
                    remainder = hyperparameters.num_modes % num_

                    aux_indeces = np.arange(num_)
                    aux_indeces = np.tile(aux_indeces,quotient)

                    if remainder != 0:
                        aux_indeces_ = np.arange(remainder)
                        aux_indeces = np.hstack((aux_indeces,aux_indeces_))
                    
                    assert len(aux_indeces) == hyperparameters.num_modes, "Number of centerlines does not match the number of required modes"
                    aux_indeces_list.append(aux_indeces)

            centerlines_indeces = np.array(aux_indeces_list)

            relevant_centerlines = []
            for mode in range(hyperparameters.num_modes):

                # Dynamic map info (one centerline per iteration -> Max Num_Modes iterations)

                current_centerlines_indeces = centerlines_indeces[:,mode]
                current_centerlines_list = []
                for i in range(batch_size):
                    current_centerline = phy_info[i]["relevant_centerlines_abs"][current_centerlines_indeces[i]]
                    current_centerlines_list.append(current_centerline)

                current_centerlines = torch.from_numpy(np.array(current_centerlines_list)).type(torch.float).cuda(obs_traj.device)
                relevant_centerlines.append(current_centerlines.unsqueeze(0))

            relevant_centerlines = torch.cat(relevant_centerlines,dim=0)

            pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=plausible_area, relevant_centerlines=relevant_centerlines)
        else:
            pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=phy_info, relevant_centerlines=relevant_centerlines)

        if hyperparameters.output_single_agent:
            pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx,:])
        else:
            pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1])

        # Take (if specified) data of only the AGENT of interest

        if hyperparameters.output_single_agent:
            obs_traj = obs_traj[:,agent_idx, :]
            pred_traj_gt = pred_traj_gt[:, agent_idx, :]
            obs_traj_rel = obs_traj_rel[:, agent_idx, :]

        # TODO: Check if in other configurations the losses are computed using relative or absolute
        # coordinates

        losses = {}

        if hyperparameters.loss_type_g == "mse" or hyperparameters.loss_type_g == "mse_w":

            if "mse_w" in hyperparameters.loss_type_g:   
                _, num_objs, _ = pred_traj_gt.shape
                w_loss_ = w_loss[:num_objs,:]
                loss_ade, loss_fde = calculate_mse_gt_loss_multimodal(pred_traj_gt, pred_traj_fake, loss_f, w_loss=w_loss_)
            else:
                loss_ade, loss_fde = calculate_mse_gt_loss_multimodal(pred_traj_gt, pred_traj_fake, loss_f)

            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()

        elif hyperparameters.loss_type_g == "nll":
            loss = calculate_nll_loss(pred_traj_gt, pred_traj_fake,loss_f)
            losses["G_nll_loss"] = loss.item()

        elif hyperparameters.loss_type_g == "mse+fa" or hyperparameters.loss_type_g == "mse_w+fa":
            loss_ade, loss_fde = calculate_mse_gt_loss_multimodal(pred_traj_gt, pred_traj_fake, loss_f["mse"])
            loss_fa = evaluate_feasible_area_prediction(pred_traj_fake, pred_traj_gt, map_origin, num_seq, 
                                                        absolute_root_folder, split)

            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde + \
                   hyperparameters.loss_fa_weight*loss_fa 

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_fa_loss"] = loss_fa.item()

        elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":

            if "mse_w" in hyperparameters.loss_type_g:
                _, num_objs, _ = pred_traj_gt.shape
                w_loss_ = w_loss[:num_objs,:]
                loss_ade, loss_fde = calculate_mse_gt_loss_multimodal(pred_traj_gt, pred_traj_fake, loss_f["mse"], w_loss=w_loss_)
                # if torch.is_tensor(relevant_centerlines):
                #     loss_ade_centerlines = calculate_mse_centerlines_loss(relevant_centerlines, pred_traj_fake, loss_f["mse"], w_loss)
            else:
                loss_ade, loss_fde = calculate_mse_gt_loss_multimodal(pred_traj_gt, pred_traj_fake, loss_f["mse"])
                # if torch.is_tensor(relevant_centerlines):
                #     loss_ade_centerlines = calculate_mse_centerlines_loss(relevant_centerlines, pred_traj_fake, loss_f["mse"])

            # _, loss_fde_goal = calculate_mse_gt_loss_multimodal(phy_info.permute(1,0,2), pred_traj_fake, loss_f["mse"], compute_ade=False)

            loss_nll = calculate_nll_loss(pred_traj_gt, pred_traj_fake, loss_f["nll"], conf)
            
            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde + \
                   hyperparameters.loss_nll_weight*loss_nll #+ \
                #    0.5*loss_fde_goal
                #    hyperparameters.loss_ade_centerlines_weight*loss_ade_centerlines 

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_nll_loss"] = loss_nll.item()
            # losses["G_mse_fde_goal_loss"] = loss_fde_goal.item()
            # losses["G_mse_ade_centerlines_loss"] = loss_nll.item()
        elif hyperparameters.loss_type_g == "centerlines+gt":

            loss_ade_centerlines, loss_fde_centerlines = calculate_mse_centerlines_loss(relevant_centerlines, pred_traj_fake, loss_f["mse"])
            
            best_pred_traj_fake_indeces = conf.argmax(1)
            best_pred_traj_fake = get_best_predictions(pred_traj_fake,best_pred_traj_fake_indeces)

            loss_ade_gt, loss_fde_gt = calculate_mse_gt_loss_unimodal(pred_traj_gt, best_pred_traj_fake, loss_f["mse"])
            
            loss = loss_ade_centerlines + \
                   loss_fde_centerlines + \
                   2*loss_ade_gt + \
                   2*loss_fde_gt

            losses["G_mse_ade_centerlines_loss"] = loss_ade_centerlines.item()
            losses["G_mse_fde_centerlines_loss"] = loss_fde_centerlines.item()
            losses["G_mse_ade_gt_loss"] = loss_ade_gt.item()
            losses["G_mse_fde_gt_loss"] = loss_fde_gt.item()

        losses[f"G_total_loss"] = loss.item()

    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )

    if USE_SCALER:
        scaler.scale(loss).backward()
        scaler.step(optimizer_g)
        scaler.update()
    else:
        loss.backward()
        optimizer_g.step()

    return losses

def check_accuracy(hyperparameters, loader, generator, 
                   limit=False, split="train"):
    """
    """
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0

    # Set generator to eval mode (not training now)

    generator.eval()

    with torch.no_grad(): # Do not compute the gradients (only when we want to check the accuracy)
        for batch in loader:
            # Load data in device

            if hyperparameters.physical_context != "plausible_centerlines+area":
                # Here the physical info is a single tensor

                batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
                 loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm, phy_info) = batch
            elif hyperparameters.physical_context == "plausible_centerlines+area":

                # TODO: In order to improve this, seq_collate should return a dictionary instead of a tuple
                # in order to avoid hardcoded positions
                phy_info = batch[-1] # phy_info should be in the last position!

                batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
                loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm) = batch

            batch_size = seq_start_end.shape[0]

            # Take (if specified) data of only the AGENT of interest

            agent_idx = None
            if hyperparameters.output_single_agent:
                agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # Forward (If agent_idx != None, model prediction refers only to target agent)

            if hyperparameters.physical_context == "plausible_centerlines+area":
                # TODO: Encapsulate this as a function

                # Plausible area (discretized)

                plausible_area_array = np.array([curr_phy_info["plausible_area_abs"] for curr_phy_info in phy_info]).astype(np.float32)
                plausible_area = torch.from_numpy(plausible_area_array).type(torch.float).cuda(obs_traj.device)

                # Compute relevant centerlines

                num_relevant_centerlines = [len(curr_phy_info["relevant_centerlines_abs"]) for curr_phy_info in phy_info]
                aux_indeces_list = []
                for num_ in num_relevant_centerlines:
                    if num_ >= hyperparameters.num_modes: # Limit to the first hyperparameters.num_modes centerlines. We assume
                                                          # the oracle is here. TODO: Force this from the preprocessing 
                        aux_indeces = np.arange(hyperparameters.num_modes)
                        aux_indeces_list.append(aux_indeces)
                    else: # We have less centerlines than hyperparameters.num_modes
                        quotient = int(hyperparameters.num_modes / num_)
                        remainder = hyperparameters.num_modes % num_

                        aux_indeces = np.arange(num_)
                        aux_indeces = np.tile(aux_indeces,quotient)

                        if remainder != 0:
                            aux_indeces_ = np.arange(remainder)
                            aux_indeces = np.hstack((aux_indeces,aux_indeces_))
                        
                        assert len(aux_indeces) == hyperparameters.num_modes, "Number of centerlines does not match the number of required modes"
                        aux_indeces_list.append(aux_indeces)
                
                centerlines_indeces = np.array(aux_indeces_list)

                relevant_centerlines = []
                for mode in range(hyperparameters.num_modes):

                    # Dynamic map info (one centerline per iteration -> Max Num_Modes iterations)

                    current_centerlines_indeces = centerlines_indeces[:,mode]
                    current_centerlines_list = []
                    for i in range(batch_size):
                        current_centerline = phy_info[i]["relevant_centerlines_abs"][current_centerlines_indeces[i]]
                        current_centerlines_list.append(current_centerline)

                    current_centerlines = torch.from_numpy(np.array(current_centerlines_list)).type(torch.float).cuda(obs_traj.device)
                    relevant_centerlines.append(current_centerlines.unsqueeze(0))

                relevant_centerlines = torch.cat(relevant_centerlines,dim=0)

                pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=plausible_area, relevant_centerlines=relevant_centerlines)
            else:
                pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=phy_info, relevant_centerlines=None)

            if hyperparameters.output_single_agent:
                pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx,:])
            else:
                pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1])

            # Mask and linear

            if not hyperparameters.output_single_agent: 
                mask = np.where(obj_id.cpu() == -1, 0, 1)
                mask = torch.tensor(mask, device=obj_id.device).reshape(-1)

            if hyperparameters.output_single_agent:
                non_linear_obj = non_linear_obj[agent_idx]
                loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj
            else:
                loss_mask = loss_mask[:, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj

            # Single agent trajectories

            if hyperparameters.output_single_agent:
                obs_traj = obs_traj[:,agent_idx, :]
                pred_traj_gt = pred_traj_gt[:, agent_idx, :]
                obs_traj_rel = obs_traj_rel[:, agent_idx, :]
                pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]

            # Relative displacements to absolute (around center) coordinates. 
            # NOT global (map) coordinates

            # L2 loss

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(pred_traj_gt.permute(1,0,2), pred_traj_gt_rel.permute(1,0,2), 
                                                         pred_traj_fake, pred_traj_fake_rel, loss_mask)

            # Evaluation metrics

            ade, ade_min = cal_ade_multimodal(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            fde, fde_min = cal_fde_multimodal(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade_min.sum())
            f_disp_error.append(fde_min.sum())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_obj).item()
            total_traj_nl += torch.sum(non_linear_obj).item()
            if limit and total_traj >= hyperparameters.num_samples_check:
                break

    metrics[f'{split}_g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics[f'{split}_g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics[f'{split}_ade'] = sum(disp_error) / (total_traj * hyperparameters.pred_len)
    metrics[f'{split}_fde'] = sum(f_disp_error) / total_traj

    # Not used at this moment

    if total_traj_l != 0:
        metrics[f'{split}_ade_l'] = sum(disp_error_l) / (total_traj_l * hyperparameters.pred_len)
        metrics[f'{split}_fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics[f'{split}_ade_l'] = 0
        metrics[f'{split}_fde_l'] = 0

    if total_traj_nl != 0:
        metrics[f'{split}_ade_nl'] = sum(disp_error_nl) / (total_traj_nl * hyperparameters.pred_len)
        metrics[f'{split}_fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics[f'{split}_ade_nl'] = 0
        metrics[f'{split}_fde_nl'] = 0

    # Not used at this moment

    # Set generator to train mode again

    generator.train()

    return metrics