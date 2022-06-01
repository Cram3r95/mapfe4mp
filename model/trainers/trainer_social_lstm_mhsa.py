import argparse
import gc
import logging
import os
import sys
import time
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
from torch.cuda.amp import GradScaler, autocast 

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
from model.models.social_lstm_mhsa import TrajectoryGenerator
from model.modules.losses import gan_g_loss, l2_loss, gan_g_loss_bce, pytorch_neg_multi_log_likelihood_batch, mse_custom
from model.modules.evaluation_metrics import displacement_error, final_displacement_error
from model.datasets.argoverse.dataset_utils import relative_to_abs_sgan
from model.utils.checkpoint_data import Checkpoint, get_total_norm
from model.utils.utils import create_weights

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
scaler = GradScaler()

current_cuda = None

# single agent False -> does not work

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(use_gpu):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def handle_batch(batch, is_single_agent_out):
    # load batch in cuda
    batch = [tensor.cuda(current_cuda) for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list) = batch
    
    # handle single agent 
    agent_idx = None
    if is_single_agent_out: # search agent idx
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
        pred_traj_gt = pred_traj_gt[:,agent_idx, :]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]

    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list)

def calculate_nll_loss(gt, pred, loss_f):
    time, bs, _ = pred.shape
    gt = gt.permute(1,0,2)
    pred = pred.contiguous().unsqueeze(1).permute(2,1,0,3)
    confidences = torch.ones(bs,1).cuda(current_cuda)
    avails = torch.ones(bs,time).cuda(current_cuda)
    loss = loss_f(
        gt, 
        pred,
        confidences,
        avails
    )
    return loss

def calculate_mse_loss(gt, pred, loss_f, l_type):
    loss_ade = loss_f(pred, gt)
    loss_fde = loss_f(pred[-1].unsqueeze(0), gt[-1].unsqueeze(0))
    return loss_ade, loss_fde

def model_trainer(config, logger):
    """
    """

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global current_cuda
    current_cuda = torch.device(f"cuda:{config.device_gpu}")
    device = torch.device(current_cuda if torch.cuda.is_available() else "cpu")
    
    long_dtype, float_dtype = get_dtypes(config.use_gpu)

    logger.info('Configuration: ')
    logger.info(config)

    logger.info("Initializing train dataset") 
    data_train = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                   root_folder=config.dataset.path,
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split="train",
                                                   split_percentage=config.dataset.split_percentage,
                                                   shuffle=config.dataset.shuffle,
                                                   batch_size=config.dataset.batch_size,
                                                   class_balance=config.dataset.class_balance,
                                                   obs_origin=config.hyperparameters.obs_origin,
                                                   preprocess_data=config.dataset.preprocess_data,
                                                   save_data=config.dataset.save_data)

    train_loader = DataLoader(data_train,
                              batch_size=config.dataset.batch_size,
                              shuffle=config.dataset.shuffle,
                              num_workers=config.dataset.num_workers,
                              collate_fn=seq_collate)

    logger.info("Initializing val dataset")
    data_val = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                 root_folder=config.dataset.path,
                                                 obs_len=config.hyperparameters.obs_len,
                                                 pred_len=config.hyperparameters.pred_len,
                                                 distance_threshold=config.hyperparameters.distance_threshold,
                                                 split="val",
                                                 split_percentage=config.dataset.split_percentage,
                                                 shuffle=config.dataset.shuffle,
                                                 class_balance=-1,
                                                 obs_origin=config.hyperparameters.obs_origin,
                                                 preprocess_data=config.dataset.preprocess_data,
                                                 save_data=config.dataset.save_data)
    val_loader = DataLoader(data_val,
                            batch_size=config.dataset.batch_size,
                            shuffle=config.dataset.shuffle,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)


    hyperparameters = config.hyperparameters
    optim_parameters = config.optim_parameters

    iterations_per_epoch = len(data_train) / config.dataset.batch_size
    if hyperparameters.num_epochs:
        hyperparameters.num_iterations = int(iterations_per_epoch * hyperparameters.num_epochs)
        hyperparameters.num_iterations = hyperparameters.num_iterations if hyperparameters.num_iterations != 0 else 1
    

    logger.info(
        'There are {} iterations per epoch'.format(hyperparameters.num_iterations)
    )

    # generator = TrajectoryGenerator(h_dim=128)
    generator = TrajectoryGenerator(h_dim=config.model.generator.hdim,current_cuda=current_cuda)
    generator.to(device)
    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Generator model:')
    logger.info(generator)

    # optimizer, scheduler and loss functions

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

    optimizer_g = optim.Adam(generator.parameters(), lr=optim_parameters.g_learning_rate, weight_decay=optim_parameters.g_weight_decay)
    if hyperparameters.lr_schduler:
        # scheduler_g = lrs.ExponentialLR(optimizer_g, gamma=hyperparameters.lr_scheduler_gamma_g)
        scheduler_g = lrs.ReduceLROnPlateau(
            optimizer_g, "min", min_lr=1e-6, verbose=True, factor=0.5, patience=15000,
        )

    restore_path = None
    if hyperparameters.checkpoint_start_from is not None:
        restore_path = hyperparameters.checkpoint_start_from
    elif hyperparameters.restore_from_checkpoint == 1:
        restore_path = os.path.join(hyperparameters.output_dir,
                                    '%s_with_model.pt' % hyperparameters.checkpoint_name)


    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint.config_cp['g_best_state'], strict=False)
        optimizer_g.load_state_dict(checkpoint.config_cp['g_optim_state'])
        # t = checkpoint.config_cp['counters']['t']
        # epoch = checkpoint.config_cp['counters']['epoch']
        t,epoch = 0,0
        checkpoint.config_cp['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = Checkpoint()

    if hyperparameters.tensorboard_active:
        exp_path = os.path.join(
            config.base_dir, hyperparameters.output_dir, "tensorboard_logs"
        )
        os.makedirs(exp_path, exist_ok=True)
        writer = SummaryWriter(exp_path)

    logger.info(f"Train {len(train_loader)}")
    logger.info(f"Val {len(val_loader)}")

    ## start training
    while t < hyperparameters.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader: # bottleneck

            losses_g = generator_step(hyperparameters, batch, generator,
                                        optimizer_g, loss_f, w_loss)
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

            if t > 0 and t % hyperparameters.checkpoint_train_every == 0:
                metrics_train = check_accuracy(
                    hyperparameters, train_loader, generator
                )

                for k, v in sorted(metrics_train.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["metrics_train"].keys():
                        checkpoint.config_cp["metrics_train"][k] = []
                    checkpoint.config_cp["metrics_train"][k].append(v)

            if t > 0 and t % hyperparameters.checkpoint_every == 0:
                checkpoint.config_cp["counters"]["t"] = t
                checkpoint.config_cp["counters"]["epoch"] = epoch
                checkpoint.config_cp["sample_ts"].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                # TODO add trainer metrics -> Compare for overfitting/underfitting

                

                metrics_val = check_accuracy(
                    hyperparameters, val_loader, generator
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["metrics_val"].keys():
                        checkpoint.config_cp["metrics_val"][k] = []
                    checkpoint.config_cp["metrics_val"][k].append(v)

                min_ade = min(checkpoint.config_cp["metrics_val"]['ade'])
                min_fde = min(checkpoint.config_cp["metrics_val"]['fde'])
                min_ade_nl = min(checkpoint.config_cp["metrics_val"]['ade_nl'])
                logger.info("Min ADE: {}".format(min_ade))
                logger.info("Min FDE: {}".format(min_fde))
                if metrics_val['ade'] <= min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint.config_cp["best_t"] = t
                    checkpoint.config_cp["g_best_state"] = generator.state_dict()

                if metrics_val['ade_nl'] <= min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint.config_cp["best_t_nl"] = t
                    checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                if metrics_val['ade'] <= min_ade:
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
        
            if hyperparameters.lr_schduler:
                scheduler_g.step(losses_g["G_total_loss"])
                g_lr = get_lr(optimizer_g)
                writer.add_scalar("G_lr", g_lr, epoch+1)
    ###
    logger.info("Training finished")

    # Check stats on the validation set
    t += 1
    epoch += 1
    checkpoint.config_cp["counters"]["t"] = t
    checkpoint.config_cp["counters"]["epoch"] = epoch+1
    checkpoint.config_cp["sample_ts"].append(t)
    logger.info('Checking stats on val ...')
    metrics_val = check_accuracy(
        hyperparameters, val_loader, generator
    )

    for k, v in sorted(metrics_val.items()):
        logger.info('  [val] {}: {:.3f}'.format(k, v))
        if hyperparameters.tensorboard_active:
            writer.add_scalar(k, v, t+1)
        if k not in checkpoint.config_cp["metrics_val"].keys():
            checkpoint.config_cp["metrics_val"][k] = []
        checkpoint.config_cp["metrics_val"][k].append(v)

    min_ade = min(checkpoint.config_cp["metrics_val"]['ade'])
    min_ade_nl = min(checkpoint.config_cp["metrics_val"]['ade_nl'])

    if metrics_val['ade'] <= min_ade:
        logger.info('New low for avg_disp_error')
        checkpoint.config_cp["best_t"] = t
        checkpoint.config_cp["g_best_state"] = generator.state_dict()

    if metrics_val['ade_nl'] <= min_ade_nl:
        logger.info('New low for avg_disp_error_nl')
        checkpoint.config_cp["best_t_nl"] = t
        checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()

    # Save another checkpoint with model weights and
    # optimizer state
    checkpoint.config_cp["g_state"] = generator.state_dict()
    checkpoint.config_cp["g_optim_state"] = optimizer_g.state_dict()
    checkpoint_path = os.path.join(
        config.base_dir, hyperparameters.output_dir, "{}_{}_with_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
    )


def generator_step(
    hyperparameters, batch, generator, optimizer_g, loss_f, w_loss=None
):
 
    batch = [tensor.cuda(current_cuda) for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

    # place holder loss
    losses = {}
    # single agent output idx
    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

    if hyperparameters.output_single_agent:
        loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
    else:  # 160x30 -> 0 o 1
        loss_mask = loss_mask[:, hyperparameters.obs_len:]

    # forward
    optimizer_g.zero_grad()
    with autocast():
        generator_out = generator(
            obs_traj, obs_traj_rel, seq_start_end, agent_idx
        )

        pred_traj_fake_rel = generator_out
        if hyperparameters.output_single_agent:
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1,agent_idx, :])
        else:
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])

        # handle single agent output
        if hyperparameters.output_single_agent:
            obs_traj = obs_traj[:,agent_idx, :]
            pred_traj_gt = pred_traj_gt[:,agent_idx, :]
            obs_traj_rel = obs_traj_rel[:, agent_idx, :]

        # calculate full traj
        # traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        # traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        # loss with relatives or abs (?) # TODO full trajectory vs pred trajectory
        if hyperparameters.loss_type_g == "mse" or hyperparameters.loss_type_g == "mse_w":
            _,b,_ = pred_traj_gt_rel.shape
            w_loss = w_loss[:b, :]
            loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt_rel, pred_traj_fake_rel, loss_f, hyperparameters.loss_type_g)
            loss = loss_ade + loss_fde
            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
        elif hyperparameters.loss_type_g == "nll":
            loss = calculate_nll_loss(pred_traj_gt_rel, pred_traj_fake_rel,loss_f)
            losses["G_nll_loss"] = loss.item()
        elif hyperparameters.loss_type_g == "mse+nll" or hyperparameters.loss_type_g == "mse_w+nll":
            _,b,_ = pred_traj_gt_rel.shape
            w_loss = w_loss[:b, :]
            loss_ade, loss_fde = calculate_mse_loss(pred_traj_gt_rel, pred_traj_fake_rel, loss_f["mse"], hyperparameters.loss_type_g)
            loss_nll = calculate_nll_loss(pred_traj_gt_rel, pred_traj_fake_rel,loss_f["nll"])
            loss = loss_ade + loss_fde*1.5 + loss_nll*0.75 
            losses["G_mse_ade_loss"] = loss_ade.item()
            losses["G_mse_fde_loss"] = loss_fde.item()
            losses["G_nll_loss"] = loss_nll.item()
        
        losses['G_total_loss'] = loss.item()

    
    scaler.scale(loss).backward()
    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )
    scaler.step(optimizer_g)
    scaler.update()
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
            batch = [tensor.cuda(current_cuda) for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

            # single agent output idx
            agent_idx = None
            if hyperparameters.output_single_agent:
                agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # mask and linear
            if not hyperparameters.output_single_agent: # TODO corregir con el nuevo dataset
                mask = np.where(obj_id.cpu() == -1, 0, 1)
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
            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end, agent_idx
            )

            # single agent trajectories
            if hyperparameters.output_single_agent:
                obs_traj = obs_traj[:,agent_idx, :]
                pred_traj_gt = pred_traj_gt[:,agent_idx, :]
                obs_traj_rel = obs_traj_rel[:, agent_idx, :]
                pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
            
            # print("pred_traj_fake_rel min ", pred_traj_fake_rel.min(), pred_traj_gt_rel.min())
            # print("pred_traj_fake_rel max ", pred_traj_fake_rel.max(), pred_traj_gt_rel.max())

            # rel to abs
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])

            # l2 loss
            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

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

def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel

def cal_ade(pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, consider_ped)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_obj)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_obj)
    return ade, ade_l, ade_nl

def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], consider_ped)
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_obj
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_obj
    )
    return fde, fde_l, fde_nl
