#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## losses

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import pdb
import os

# DL & Math imports

import torch
import random
import numpy as np
import cv2
from torch import Tensor

#######################################

# Adversarial losses

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.8, 1)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def gan_g_loss_bce(scores_fake, bce):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.8, 1)
    return bce(scores_fake, y_fake)

def gan_d_loss_bce(scores_real, scores_fake, bce):
    y_real = torch.ones_like(scores_real) * random.uniform(0.8, 1)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.2)
    loss_real = bce(scores_real, y_real)
    loss_fake = bce(scores_fake, y_fake)
    return loss_real + loss_fake

# Distance losses

def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def l2_loss_multimodal(pred_traj, pred_traj_gt, random=0, mode='average'):
    """
    gt (b,t,2)
    pred (b,m,t,2)
    """
    b,m,t,_ = pred_traj.size()
    loss = (
            (pred_traj_gt.unsqueeze(dim=1) - pred_traj)**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
   
def mse(gt, pred, w_loss=None):
    """
    With t = predicted points (pred_len)
    gt: (t, b, 2)
    pred: (t, b, 2)
    w_loss: (b, t)
    """
    try:
        l2 = gt.permute(1, 0, 2) - pred.permute(1, 0, 2) # b,t,2
        l2 = l2**2 # b, t, 2
        l2 = torch.sum(l2, axis=2) # b, t
        l2 = torch.sqrt(l2) # b, t
        
        if torch.is_tensor(w_loss):
            l2 = l2 * w_loss # b, t

        l2 = torch.mean(l2, axis=1) # b
        l2 = torch.mean(l2) # single value
    except Exception as e:
        print(e)
        pdb.set_trace()
    return l2

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor,
    pred: Tensor,
    confidences: Tensor,
    avails: Tensor,
    epsilon: float = 1.0e-8,
    is_reduce: bool = True,
) -> Tensor:
    """
    from
    https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence/comments

    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow,
    For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size,
        num_modes,
    ), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(
        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))
    ), "confidences should sum to 1"
    assert avails.shape == (
        batch_size,
        future_len,
    ), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"
    # pdb.set_trace()
    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - pred) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        # error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time
        error = torch.log(confidences + epsilon) - 0.5 * torch.sum(error, dim=-1)
    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(
        dim=1, keepdim=True
    )  # error are negative at this point, so max() gives the minimum one
    error = (
        -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True))
        - max_value
    )  # reduce modes
    # print("error", error)
    if is_reduce:
        return torch.mean(error)
    else:
        return error

def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """
    from
    https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence/comments

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(
        gt, pred.unsqueeze(1), confidences, avails
    )


def _average_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: average displacement error (ADE), a single float number
    """

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = np.mean(error, axis=-1)  # average over timesteps

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def _final_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: final displacement error (FDE), a single float number
    """

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = error[:, -1]  # use last timestep

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def l2_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """A function that takes pred, gt tensor and computes their L2 distance.
    :param pred: predicted tensor, size: [batch_size, num_dims]
    :param gt: gt tensor, size: [batch_size, num_dims]
    :return: l2 distance between the predicted and gt tensor, size: [batch_size,]
    """
    return torch.norm(pred - gt, p=2, dim=-1)

def _assert_shapes(ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> None:
    """
    Check the shapes of args required by metrics
    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps
    Returns:
    """
    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (num_modes,), f"expected 1D (Modes) array for confidences, got {confidences.shape}"
    assert np.allclose(np.sum(confidences), 1), "confidences should sum to 1"
    assert avails.shape == (future_len,), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(ground_truth).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"

def evaluate_feasible_area_prediction(pred_traj_fake_abs, pred_traj_gt_abs, map_origin, num_seq, 
                                      absolute_root_folder, split, dist_rasterized_map=[-40,40,-40,40]):
    """
    Get feasible_area_loss. If a prediction point (in pixel coordinates) is in the drivable (feasible)
    area, is weighted with 1. Otherwise, it is weighted with 0. Theoretically, most AGENT points (observation
    and prediction) must be in the Feasible Area in Argoverse 1.1.

    Input:
        pred_traj_fake_abs: Torch.tensor -> pred_len x 2 (x|y) in absolute coordinates (around 0,0)
        map_origin: Tor
        filename: Image filename to read
    Output:
        feasible_area_loss: min = 0 (num_points · 1), max = pred_len (num_points · 1)
    """

    feasible_area_loss = torch.zeros((pred_traj_fake_abs.shape[1]))
    feasible_area_loss_v2 = torch.zeros((pred_traj_fake_abs.shape[1]))

    pred_len = pred_traj_fake_abs.shape[0]
    img_resize = 600
    real_world_offset = abs(dist_rasterized_map[0])

    # Iterate over the different sequences and predictions

    for t,seq in enumerate(num_seq):
        _seq = seq.item()
        filename = os.path.join(absolute_root_folder,split,"data_images",f"{_seq}.png")

        img_map = cv2.imread(filename)
        img_map = cv2.resize(img_map, dsize=(img_resize,img_resize))
        img_map_gray = cv2.cvtColor(img_map,cv2.COLOR_BGR2GRAY)

        xcenter, ycenter = map_origin[t][0][0], map_origin[t][0][1]
        x_min = xcenter + dist_rasterized_map[0]
        x_max = xcenter + dist_rasterized_map[1]
        y_min = ycenter + dist_rasterized_map[2]
        y_max = ycenter + dist_rasterized_map[3]

        ## Compute slope and intercept to transform global point to pixel

        m_x = float(img_resize / (2 * real_world_offset)) # slope
        m_y = float(-img_resize / (2 * real_world_offset))

        i_x = float(-(img_resize / (2 * real_world_offset)) * x_min) # intercept
        i_y = float((img_resize / (2 * real_world_offset)) * y_max)

        _feasible_area_loss = []
        _feasible_area_loss_v2 = []

        for i in range(pred_len):
            # TODO: Check map tensors in new .npy save
            pix_x = int((pred_traj_fake_abs[i,t,0]+map_origin[t][0][0]) * m_x + i_x)
            pix_y = int((pred_traj_fake_abs[i,t,1]+map_origin[t][0][1]) * m_y + i_y)

            gt_pix_x = int((pred_traj_gt_abs[i,t,0]+map_origin[t][0][0]) * m_x + i_x)
            gt_pix_y = int((pred_traj_gt_abs[i,t,1]+map_origin[t][0][1]) * m_y + i_y)

            # Out of the image (we assume the physical context is big enough to capture the whole prediction)

            if pix_y > img_resize-1 or pix_y < 0 or pix_x > img_resize-1 or pix_x < 0: 
                _feasible_area_loss.append(0)
                _feasible_area_loss_v2.append(0)
            else:    
                if img_map_gray[pix_y,pix_x] == 255.0: _feasible_area_loss.append(1)
                else: _feasible_area_loss.append(0)
                if img_map_gray[pix_y,pix_x] == img_map_gray[gt_pix_y,gt_pix_x]: 
                    _feasible_area_loss_v2.append(1)
                else:
                    _feasible_area_loss_v2.append(0)

        feasible_area_loss[t] = sum(_feasible_area_loss)
        feasible_area_loss_v2[t] = sum(_feasible_area_loss_v2)

    feasible_area_loss_v2 = feasible_area_loss_v2.mean()
    #pdb.set_trace()
    return feasible_area_loss
