#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Trainer Set Transformer Multimodal

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
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
from model.models.social_set_transformer_mm import TrajectoryGenerator
from model.modules.losses import l2_loss_multimodal, mse_custom, pytorch_neg_multi_log_likelihood_batch, evaluate_feasible_area_prediction
from model.modules.evaluation_metrics import displacement_error, final_displacement_error
from model.utils.checkpoint_data import Checkpoint, get_total_norm
from model.datasets.argoverse.dataset_utils import relative_to_abs_multimodal
from model.utils.utils import create_weights

#######################################

# Global variables

torch.backends.cudnn.benchmark = True
scaler = GradScaler()
current_cuda = None
absolute_root_folder = None

MAX_TIME_TO_CHECK_TRAIN = 45 # minutes
MAX_TIME_TO_CHECK_VAL = 30 # minutes
MAX_TIME_PATIENCE_LR_SCHEDULER = 60 # minutes

# Aux functions

def get_lr(optimizer):
    """
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_weights(m):
    """
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

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

def calculate_mse_loss(gt, pred, loss_f):
    """
    MSE = Mean Square Error (it is used by both ADE and FDE)
    gt,pred: (batch_size,num_modes,pred_len,2)
    """

    b,m,t,_ = pred.shape
    pred = pred.permute(1,2,0,3) # (num_modes,pred_len,batch_size,data_dim=2)
    loss_ade = torch.zeros(1).to(pred)
    loss_fde = torch.zeros(1).to(pred)

    for i in range(m):
        loss_ade += loss_f(pred[i,:,:,:], gt)
        loss_fde += loss_f(pred[i][-1].unsqueeze(0), gt[-1].unsqueeze(0))
    return loss_ade/m, loss_fde/m
 
def calculate_nll_loss(gt, pred, loss_f, confidences):
    """
    NLL = Negative Log-Likelihood
    """
    time, bs, _ = pred.shape
    gt = gt.permute(1,0,2)
    avails = torch.ones(bs,time).cuda(current_cuda)
    loss = loss_f(
        gt, 
        pred,
        confidences,
        avails
    )
    return loss



def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel):
    """
    """
    g_l2_loss_abs = l2_loss_multimodal(pred_traj_fake, pred_traj_gt, mode='sum')
    g_l2_loss_rel = l2_loss_multimodal(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')
    
    return g_l2_loss_abs, g_l2_loss_rel

def cal_ade(pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped):
    """
    """
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

def cal_fde(pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped):
    """
    """
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
                                                 preprocess_data=config.dataset.preprocess_data,
                                                 save_data=config.dataset.save_data)

    val_loader = DataLoader(data_val,
                            batch_size=config.dataset.batch_size,
                            shuffle=False,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)

    # Initialize motion prediction generator and optimizer

    ## Compute total number of iterations (iterations_per_epoch * num_epochs)

    iterations_per_epoch = len(data_train) / config.dataset.batch_size
    if hyperparameters.num_epochs:
        hyperparameters.num_iterations = int(iterations_per_epoch * hyperparameters.num_epochs) # compute total iterations
        assert hyperparameters.num_iterations != 0

    logger.info(
        'There are {} iterations per epoch'.format(hyperparameters.num_iterations)
    )

    # Initialize motion prediction generator

    generator = TrajectoryGenerator()
    generator.to(device)
    generator.apply(init_weights)
    generator.type(float_dtype).train() # train mode (if you compute metrics -> .eval() mode)
    logger.info('Generator model:')
    logger.info(generator)

    ## Restore from checkpoint if specified in config file

    restore_path = None
    if hyperparameters.checkpoint_start_from is not None:
        restore_path = hyperparameters.checkpoint_start_from

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path, map_location=current_cuda)
        generator.load_state_dict(checkpoint.config_cp['g_best_state'], strict=False)
        optimizer_g.load_state_dict(checkpoint.config_cp['g_optim_state'])

        t = checkpoint.config_cp['counters']['t'] # Restore num_iters and epochs from checkpoint
        epoch = checkpoint.config_cp['counters']['epoch']
        # t,epoch = 0,0 # set to 0, though it starts from a specific checkpoint
                        # E.g. You have trained for 30k iterations, you take that checkpoint but
                        # you want to start from 0 iterations (only a number), but the model already
                        # has the corresponding weights after 30k iterations
        checkpoint.config_cp['restore_ts'].append(t)
    else:
        # Start from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = Checkpoint() 

    # Optimizer, scheduler and loss functions

    optimizer_g = optim.Adam(generator.parameters(), lr=optim_parameters.g_learning_rate, weight_decay=optim_parameters.g_weight_decay)
    
    if hyperparameters.lr_scheduler: # Patience: Number of epochs with no improvement after 
                                     # which learning rate will be reduced
        scheduler_g = lrs.ReduceLROnPlateau(optimizer_g, "min", min_lr=1e-6, 
                                            verbose=True, factor=0.5, patience=15000)

    if hyperparameters.loss_type_g == "mse" or hyperparameters.loss_type_g == "mse_w":
        loss_f = mse_custom
    elif hyperparameters.loss_type_g == "nll":
        loss_f = pytorch_neg_multi_log_likelihood_batch
    elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":
        loss_f = {
            "mse": mse_custom,
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

    logger.info(f"Train {len(train_loader)}")
    logger.info(f"Val {len(val_loader)}")

    ###################################
    #### Main Loop. Start training ####
    ###################################

    while t < hyperparameters.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:

            losses_g = generator_step(hyperparameters, batch, generator, optimizer_g, loss_f)

            checkpoint.config_cp["norm_g"].append(
                get_total_norm(generator.parameters())
            )

            if t % hyperparameters.print_every == 0:
                # print logger
                logger.info('t = {} / {}'.format(t + 1, hyperparameters.num_iterations))
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["G_losses"].keys():
                        checkpoint.config_cp["G_losses"][k] = [] 
                    checkpoint.config_cp["G_losses"][k].append(v)
                checkpoint.config_cp["losses_ts"].append(t)

                # Check training metrics

            if t > 0 and t % hyperparameters.checkpoint_train_every == 0:
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

            if t > 0 and t % hyperparameters.checkpoint_every == 0:
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
            if t >= hyperparameters.num_iterations:
                break

            if hyperparameters.lr_scheduler:
                scheduler_g.step(losses_g["G_total_loss"])
                g_lr = get_lr(optimizer_g)
                writer.add_scalar("G_lr", g_lr, epoch+1)

    logger.info("Training finished")

def generator_step(hyperparameters, batch, generator, optimizer_g, 
                   loss_f):
    """
    """
    # Load data in device

    batch = [tensor.cuda(current_cuda) for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

    # Take (if specified) data of only the AGENT of interest

    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

    if hyperparameters.output_single_agent:
        loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
    else:
        loss_mask = loss_mask[:, hyperparameters.obs_len:]

    # Forward

    optimizer_g.zero_grad()
    with autocast():
        generator_out, conf = generator(obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out

        if hyperparameters.output_single_agent:
            pred_traj_fake = relative_to_abs_sgan_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx, :])
        else:
            pred_traj_fake = relative_to_abs_sgan_multimodal(pred_traj_fake_rel, obs_traj[-1])
            

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
            w_loss = w_loss[:b, :]
            loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt_rel, pred_traj_fake_rel, 
                                                    loss_f["mse"])
            loss = loss_ade + loss_fde
            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
        elif hyperparameters.loss_type_g == "nll":
            loss = calculate_nll_loss(pred_traj_gt_rel, pred_traj_fake_rel,loss_f)
            losses["G_nll_loss"] = loss.item()
        elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":
            _,b,_ = pred_traj_gt_rel.shape
            loss_ade, loss_fde = calculate_mse_loss(
                pred_traj_gt_rel, pred_traj_fake_rel, loss_f["mse"]
            )
            loss_nll = calculate_nll_loss(pred_traj_gt_rel, pred_traj_fake_rel,loss_f["nll"], conf)

            loss = 1*loss_ade + 2.5*loss_fde + 1.5*loss_nll # TODO: Set these weights in config file

            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_nll_loss"] = loss_nll.item()
    
    losses['G_total_loss'] = loss.item()

    
    loss.backward() # scaler.scale(loss).backward()
    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )
    # scaler.step(optimizer_g)
    optimizer_g.step()
    # scaler.update()

    return losses

def check_accuracy(
    hyperparameters, loader, generator, limit=False
):
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

            # single agent output idx
            agent_idx = None
            if hyperparameters.output_single_agent:
                agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # mask and linear
            if not hyperparameters.output_single_agent: # TODO corregir con el nuevo dataset
                mask = np.where(obj_id.cpu() == -1, 0, 1)
                pdb.set_trace()
                mask = torch.tensor(mask, device=obj_id.device).reshape(-1)
            if hyperparameters.output_single_agent:
                # mask = mask[agent_idx]
                non_linear_obj = non_linear_obj[agent_idx]
                loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj
            else:  # 160x30 -> 0 o 1
                loss_mask = loss_mask[:, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj

            ## forward
            pred_traj_fake_rel, conf = generator(
                obs_traj_rel, seq_start_end
            )
            # rel to abs
            if hyperparameters.output_single_agent:
                pred_traj_fake = relative_to_abs_sgan_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx, :])
            else:
                pred_traj_fake = relative_to_abs_sgan_multimodal(pred_traj_fake_rel, obs_traj[-1])

            # single agent trajectories
            if hyperparameters.output_single_agent:
                obs_traj = obs_traj[:,agent_idx, :]
                pred_traj_gt = pred_traj_gt[:,agent_idx, :]
                obs_traj_rel = obs_traj_rel[:, agent_idx, :]
                pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
                # pred_traj_fake = pred_traj_fake[:, agent_idx, :]
                # pred_traj_fake_rel = pred_traj_fake_rel[:, agent_idx, :]

            # l2 loss
            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt.permute(1,0,2), pred_traj_gt_rel.permute(1,0,2), pred_traj_fake,
                pred_traj_fake_rel
            )
            ade, ade_min = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            fde, fde_min = cal_fde(
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
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * hyperparameters.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * hyperparameters.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * hyperparameters.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics
