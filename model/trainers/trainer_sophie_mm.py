#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Trainer LSTM based Encoder-Decoder with Multi-Head Self Attention

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
import copy

# DL & Math imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.tensorboard import SummaryWriter

# Custom imports

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
from model.models.sophie_mm import TrajectoryGenerator
from model.modules.losses import l2_loss_multimodal, mse, pytorch_neg_multi_log_likelihood_batch, evaluate_feasible_area_prediction
from model.modules.evaluation_metrics import displacement_error, final_displacement_error
from model.datasets.argoverse.dataset_utils import relative_to_abs_multimodal
from model.utils.checkpoint_data import Checkpoint, get_total_norm
from model.utils.utils import create_weights

#######################################

# Global variables

torch.backends.cudnn.benchmark = True
scaler = GradScaler()
current_cuda = None
absolute_root_folder = None

USE_SCALER = True
CHECK_ACCURACY_TRAIN = False
CHECK_ACCURACY_VAL = True
MAX_TIME_TO_CHECK_TRAIN = 120 # minutes
MAX_TIME_TO_CHECK_VAL = 120 # minutes
MAX_TIME_PATIENCE_LR_SCHEDULER = 120 # 60 # minutes

# Aux functions

def get_lr(optimizer):
    """
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_weights(module):
    """
    """
    classname = module.__class__.__name__
    if classname.find('Linear') != -1 and classname != "LinearRes":
        nn.init.kaiming_normal_(module.weight)

def get_dtypes(use_gpu):
    """
    """
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def handle_batch(batch, is_single_agent_out):
    """
    """
    # Load batch in cuda

    batch = [tensor.cuda(current_cuda) for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list) = batch
    
    # Handle single agent
 
    agent_idx = None
    if is_single_agent_out: # search agent idx
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
        pred_traj_gt = pred_traj_gt[:,agent_idx, :]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]

    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
            loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list)

# Aux functions losses

def calculate_mse_loss(gt, pred, loss_f, w_loss=None):
    """
    gt: (pred_len, batch_size, data_dim)
    pred: (batch_size, num_modes, pred_len, data_dim)
    """

    batch_size, num_modes, pred_len, data_dim = pred.shape
    pred = pred.permute(1,2,0,3)
    loss_ade = torch.zeros(1).to(pred)
    loss_fde = torch.zeros(1).to(pred)

    for i in range(num_modes):
        loss_ade += loss_f(pred[i,:,:,:], gt, w_loss)
        loss_fde += loss_f(pred[i][-1].unsqueeze(0), gt[-1].unsqueeze(0), w_loss)

    return loss_ade/num_modes, loss_fde/num_modes

def calculate_nll_loss(gt, pred, loss_f, confidences):
    """
    NLL = Negative Log-Likelihood
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
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split="train",
                                                   split_percentage=config.dataset.split_percentage,
                                                   batch_size=config.dataset.batch_size,
                                                   class_balance=config.dataset.class_balance,
                                                   obs_origin=config.hyperparameters.obs_origin,
                                                   data_augmentation=config.dataset.data_augmentation,
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
                                                 obs_len=config.hyperparameters.obs_len,
                                                 pred_len=config.hyperparameters.pred_len,
                                                 distance_threshold=config.hyperparameters.distance_threshold,
                                                 split="val",
                                                 split_percentage=config.dataset.split_percentage,
                                                 class_balance=-1,
                                                 obs_origin=config.hyperparameters.obs_origin,
                                                 extra_data_train=config.dataset.extra_data_train,
                                                 preprocess_data=config.dataset.preprocess_data,
                                                 save_data=config.dataset.save_data)
                                     
    val_loader = DataLoader(data_val,
                            batch_size=config.dataset.batch_size,
                            shuffle=False,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)

    # Initialize motion prediction generator and optimizer

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

    # Scheduler and loss functions
    
    if hyperparameters.lr_scheduler: 
        lr_scheduler_patience = int(hyperparameters.lr_epoch_percentage_patience * hyperparameters.num_iterations) # Every N epochs
        scheduler_g = lrs.ReduceLROnPlateau(optimizer_g, "min", min_lr=1e-5, 
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
    else:
        assert 1 == 0, "loss_type_g is not correct"

    w_loss = create_weights(config.dataset.batch_size, 1, 8).cuda(current_cuda)

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
    
    time_iterations = float(0)
    time_per_iteration = float(0)
    current_iteration = 0 # Current iteration, regardless the model had previous training
    flag_check_every = False

    while t < hyperparameters.num_iterations + previous_t:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))

        for batch in train_loader:
            start = time.time()
            losses_g = generator_step(hyperparameters, batch, generator, optimizer_g, loss_f, w_loss)
            end = time.time()
            checkpoint.config_cp["norm_g"].append(get_total_norm(generator.parameters()))
            
            time_iterations += end-start
            time_per_iteration = time_iterations / (current_iteration+1)

            # Compute approximate time to compute train and val metrics

            if not flag_check_every:
                seconds_to_check_train = time_per_iteration*hyperparameters.checkpoint_train_every
                seconds_to_check_val = time_per_iteration*hyperparameters.checkpoint_val_every
                seconds_lr_patience = time_per_iteration*lr_scheduler_patience

                if seconds_to_check_train > MAX_TIME_TO_CHECK_TRAIN*60:
                    hyperparameters.checkpoint_train_every = round((MAX_TIME_TO_CHECK_TRAIN*60) / time_per_iteration)
                if seconds_to_check_val > MAX_TIME_TO_CHECK_VAL*60:
                    hyperparameters.checkpoint_val_every = round((MAX_TIME_TO_CHECK_VAL*60) / time_per_iteration)
                if seconds_lr_patience > MAX_TIME_PATIENCE_LR_SCHEDULER*60:
                    lr_scheduler_patience = round((MAX_TIME_PATIENCE_LR_SCHEDULER*60) / time_per_iteration)

                    scheduler_g = lrs.ReduceLROnPlateau(optimizer_g, "min", min_lr=1e-5, 
                                                        verbose=True, factor=hyperparameters.lr_decay_factor, 
                                                        patience=lr_scheduler_patience)
                                                
                mins_to_check_train = round((time_per_iteration*hyperparameters.checkpoint_train_every)/60)
                mins_to_check_val = round((time_per_iteration*hyperparameters.checkpoint_val_every)/60)
                mins_to_complete = round((time_per_iteration*hyperparameters.num_iterations)/60)
                mins_to_decrease_lr = round((time_per_iteration*lr_scheduler_patience)/60)

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

            # Print losses info
            
            if current_iteration % hyperparameters.print_every == 0:
                logger.info('Iteration = {} / {}'.format(t + 1, hyperparameters.num_iterations + previous_t))
                logger.info('Time per iteration: {}'.format(time_per_iteration))

                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["G_losses"].keys():
                        checkpoint.config_cp["G_losses"][k] = [] 
                    checkpoint.config_cp["G_losses"][k].append(v)
                checkpoint.config_cp["losses_ts"].append(t)

            # Check training metrics

            if (CHECK_ACCURACY_TRAIN and current_iteration > 0 
               and current_iteration % hyperparameters.checkpoint_train_every == 0):

                logger.info('Checking stats on train ...')
                split = "train"

                metrics_train = check_accuracy(
                    hyperparameters, train_loader, generator, split=split
                )

                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["metrics_train"].keys():
                        checkpoint.config_cp["metrics_train"][k] = []
                    checkpoint.config_cp["metrics_train"][k].append(v)

            # Check validation metrics

            if (CHECK_ACCURACY_VAL and current_iteration > 0 
               and current_iteration % hyperparameters.checkpoint_val_every == 0):

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
                        writer.add_scalar(k, v, t+1)
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

                # If val_min_ade < min_ade, save another checkpoint with model weights and
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

            t += 1
            current_iteration += 1

            if current_iteration >= hyperparameters.num_iterations:
                break
        
            if hyperparameters.lr_scheduler:
                scheduler_g.step(losses_g["G_total_loss"])
                g_lr = get_lr(optimizer_g)
                writer.add_scalar("G_lr", g_lr, t+1)

    logger.info("Training finished")

    # Check train and validation metrics once the training is finished
    # Because you check every N iterations, but not specifically the last one

    ## Check train metrics

    logger.info('Checking stats on train ...')
    split = "train"

    metrics_train = check_accuracy(
        hyperparameters, train_loader, generator, split=split
    )

    for k, v in sorted(metrics_train.items()):
        logger.info('  [train] {}: {:.3f}'.format(k, v))
        if hyperparameters.tensorboard_active:
            writer.add_scalar(k, v, t+1)
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
            writer.add_scalar(k, v, t+1)
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

    batch = [tensor.cuda(current_cuda) for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, map_origin, num_seq, norm) = batch

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

        pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx)

        if hyperparameters.output_single_agent:
            pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx,:])
        else:
            pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1])

        # Take (if specified) data of only the AGENT of interest

        if hyperparameters.output_single_agent:
            obs_traj = obs_traj[:,agent_idx, :]
            pred_traj_gt = pred_traj_gt[:,agent_idx, :]
            obs_traj_rel = obs_traj_rel[:, agent_idx, :]

        # TODO: Check if in other configurations the losses are computed using relative or absolute
        # coordinates

        losses = {}

        if hyperparameters.loss_type_g == "mse" or hyperparameters.loss_type_g == "mse_w":
            _,b,_ = pred_traj_gt_rel.shape

            if hyperparameters.loss_type_g == "mse_w":
                w_loss = w_loss[:b, :]
                loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt, pred_traj_fake, loss_f, w_loss)
            else:
                loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt, pred_traj_fake, loss_f)

            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()

        elif hyperparameters.loss_type_g == "nll":
            loss = calculate_nll_loss(pred_traj_gt, pred_traj_fake,loss_f)
            losses["G_nll_loss"] = loss.item()

        elif hyperparameters.loss_type_g == "mse+fa" or hyperparameters.loss_type_g == "mse_w+fa":
            loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt, pred_traj_fake, loss_f["mse"])
            loss_fa = evaluate_feasible_area_prediction(pred_traj_fake, pred_traj_gt, map_origin, num_seq, 
                                                        absolute_root_folder, split)
            pdb.set_trace()
            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde + \
                   hyperparameters.loss_fa_weight*loss_fa 
            pdb.set_trace()
            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_fa_loss"] = loss_fa.item()

        elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":
            _,b,_ = pred_traj_gt_rel.shape

            if hyperparameters.loss_type_g == "mse_w":
                w_loss = w_loss[:b, :]
                loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt, pred_traj_fake, loss_f["mse"], w_loss)
            else:
                loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt, pred_traj_fake, loss_f["mse"])

            loss_nll = calculate_nll_loss(pred_traj_gt, pred_traj_fake, loss_f["nll"], conf)

            loss = hyperparameters.loss_ade_weight*loss_ade + \
                   hyperparameters.loss_fde_weight*loss_fde + \
                   hyperparameters.loss_nll_weight*loss_nll 

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_nll_loss"] = loss_nll.item()
        
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
            batch = [tensor.cuda(current_cuda) for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

            # Take (if specified) data of only the AGENT of interest

            agent_idx = None
            if hyperparameters.output_single_agent:
                agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

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

            # Forward (If agent_idx != None, model prediction refers only to target agent)

            pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx)

            if hyperparameters.output_single_agent:
                pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx,:])
            else:
                pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1])

            # Single agent trajectories

            if hyperparameters.output_single_agent:
                obs_traj = obs_traj[:,agent_idx, :]
                pred_traj_gt = pred_traj_gt[:,agent_idx, :]
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