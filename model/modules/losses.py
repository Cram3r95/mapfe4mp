#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## losses

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import os

# DL & Math imports

import math
import torch
import torch.nn as nn
import torch.cuda
import random
import numpy as np
import cv2
import torch.nn.functional as F

from torch.autograd import Function

from numba import jit, prange, cuda
from torch import Tensor

#######################################

smooth_l1_loss = nn.SmoothL1Loss(reduction="none") # mean, sum, none

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

def smoothL1(gt, pred):
    """
    With t = predicted points (pred_len)
    gt: (t, b, 2)
    pred: (t, b, 2)
    """

    try:
        smoothL1_ = F.smooth_l1_loss(gt,pred) # reduction = "mean"

    except Exception as e:
        print(e)
        pdb.set_trace()
    return smoothL1_

def l1_ewta_loss(prediction, target, k=6):
    """_summary_

    Args:
        prediction (_type_): _description_
        target (_type_): _description_
        k (int, optional): _description_. Defaults to 6.

    Returns:
        _type_: _description_
    """
    num_mixtures = prediction.shape[1]

    target = target.unsqueeze(1).expand(-1, num_mixtures, -1, -1)
    l1_loss = nn.functional.l1_loss(prediction, target, reduction='none').sum(dim=[2, 3])

    # Get loss from top-k mixtures for each timestep
    mixture_loss_sorted, mixture_ranks = torch.sort(l1_loss, descending=False)
    mixture_loss_topk = mixture_loss_sorted.narrow(1, 0, k)

    # Aggregate loss across timesteps and batch
    loss = mixture_loss_topk.sum()
    loss = loss / target.size(0)
    loss = loss / target.size(2)
    loss = loss / k
    return loss

def l1_wta_loss(prediction, target, conf=None):
    """_summary_

    Args:
        prediction (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
 
    num_modes = prediction.shape[1]

    target_ = target.unsqueeze(1)
    target_ = torch.repeat_interleave(target_, num_modes, dim=1)
    
    ## Reduction = Mean
    
    # loss_single = smooth_l1_loss(prediction, target_)
    # loss_out = loss_single
    
    # Reduction = None
    
    
    # bs x pred_len x num_modes x 2
    
    
    loss_single = smooth_l1_loss(prediction, target_) # bs x num_modes x pred_len x data_dim
    loss_single = torch.sum(torch.sum(loss_single, dim=3), dim=2) 
    # pdb.set_trace()
    if torch.is_tensor(conf):
        loss_single = torch.mul(loss_single,conf) # Element-wise multiplication
        loss_single = torch.sum(loss_single,axis=1) # Sum along num_modes
        loss_out = torch.mean(loss_single)
    else:
        min_loss_index = torch.argmin(loss_single, dim=1)
        min_loss_combined = [x[min_loss_index[i]] for i, x in enumerate(loss_single)]
        loss_out = torch.sum(torch.stack(min_loss_combined))

    return loss_out

def hinge_loss(conf):
    """_summary_
    """
    
    return loss

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
    try:
        assert torch.isfinite(pred).all(), "invalid value found in pred"
        assert torch.isfinite(gt).all(), "invalid value found in gt"
        assert torch.isfinite(confidences).all(), "invalid value found in confidences"
        assert torch.isfinite(avails).all(), "invalid value found in avails"
    except:
        pdb.set_trace()
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

# SoftDWT

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)
