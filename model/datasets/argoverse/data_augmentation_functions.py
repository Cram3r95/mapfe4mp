#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Data augmentation functions

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import random
import math
import pdb
import copy

# DL & Math imports

import numpy as np
import torch
import cv2
from sklearn import linear_model

# Plot imports

import matplotlib.pyplot as plt

#######################################

def get_data_aug_combinations(num_augs):
    """
    Tuple to enable (1) a given augmentation
    """

    return np.random.randint(2,size=num_augs).tolist()

def get_pairs(percentage,num_obs,start_from=1):
    """
    Return round(percentage*num_obs) non-consecutive indeces in the range(start_from,num_obs-1)
    N.B. Considering this algorithm, the maximum number to calculate non-consecutive indeces
    is 40 % (0.4)
    
    Input:
        - percentage
        - num_obs
        - start_from
    Output:
        - indeces
    """

    indeces = set()
    while len(indeces) !=  round(percentage*num_obs):
        a = random.randint(start_from,num_obs-1) # From second point (1) to num_obs - 1
        if not {a-1,a,a+1} & indeces: # Set intersection: empty == False == no common numbers
            indeces.add(a)

    indeces = sorted(indeces)
    return indeces

def swap_points(traj,num_obs=20,percentage=0.2,post=False):
    """
    Swapping (x(i) <-> x(i+1)) of observation data (Not the whole sequence)
    N.B. Points to be swapped cannot be consecutive to avoid:
    x(i),x(i+1),x(i+2) -> x(i+1),x(i),x(i+2) -> x(i+1),x(i+2),x(i) # TODO: Try this?
    Input:
        - traj: Whole trajectory (2 (x|y) x seq_len)
        - num_obs: Number of observations of the whole trajectory
        - percentage: Ratio of pairs to be swapped. Typically the number of pairs 
          to be swapped will be 0.2. E.g. N=4 if num_observations is 20
        - post: Flag to swap with the next (x(i+1)) or previous (x(i-1)) point. 
          By default: False (swap with the previous point)
    Output:
        - swapped_traj: Whole trajectory (2 (x|y) x seq_len) with swapped non-consecutive points in
          the range (1,num_obs-1)
    """

    swapped_traj = copy.deepcopy(traj)

    swapped_pairs = get_pairs(percentage,num_obs)

    for index_pair in swapped_pairs:
        aux = copy.copy(swapped_traj[:,index_pair])
        swapped_traj[:,index_pair] = swapped_traj[:,index_pair-1]
        swapped_traj[:,index_pair-1] = aux

    return swapped_traj

def dropout_points(traj,apply_dropout,num_obs=20,percentage=0.2,post=False):
    """
    Remove (x(i) and subsitute with x(i-1)) 
    E.g. x(0), x(1), x(2) -> x(0), x(0), x(2)
    Flag: Substitute with i-1 or i+1
    Input:
        - traj: 20 x num_agents x 2
        - num_obs: Number of observations of the whole trajectory
        - percentage: Ratio of pairs to be erased. Typically the number of pairs 
          to be swapped will be 0.2. E.g. N=4 if num_observations is 20
        - post: Flag to erase the current point and substitute with the next (x(i+1)) or 
          previous (x(i-1)) point. By default: False (substitute with the previous point)
    Output:
        - erased_traj: Whole trajectory (20 x num_agents x 2) with erased non-consecutive points in
          the range (1,num_obs-1)
    """

    erased_traj = copy.deepcopy(traj)
    erased_traj_aux = torch.zeros((erased_traj.shape))

    for i,app_dropout in enumerate(apply_dropout):
        
        _erased_traj = erased_traj[:,i,:]

        if app_dropout:
            erased_pairs = get_pairs(percentage,num_obs)
            for index_pair in erased_pairs:
                _erased_traj[index_pair,:] = _erased_traj[index_pair-1,:]

        erased_traj_aux[:,i,:] = _erased_traj

    return erased_traj_aux

def add_gaussian_noise(traj,apply_gaussian_noise,num_agents,num_obs=20,multi_point=True,mu=0,sigma=0.5):
    """
    Input:
        - traj: 20 x num_agents x 2
    Output: 
        - noise_traj: 20 x num_agents x 2 with gaussian noise in the observation points
    If multi_point = False, apply a single x|y offset to all observation points.
    Otherwise, apply a particular x|y per observation point.
    By default, multi_point = True since it is more challenging.
    """

    data_dim = 2
    
    if multi_point:
        size = (num_obs,num_agents,data_dim)
    else:
        size = (1,num_agents,data_dim)
    
    offset = torch.normal(mu,sigma,size=size)
    
    # Not apply in the following objects

    indeces = torch.where(apply_gaussian_noise == 0)[0]
    offset[:,indeces,:] = 0
    traj += offset

    return traj

def rotate_traj(traj,R):
    """
    traj: torch.tensor
    angle: torch.tensor in radians
    """

    # traj = torch.matmul(traj,R) # (N x 2) x (2 x 2)
    traj = np.matmul(traj,R).float() # (N x 2) x (2 x 2)
    
    return traj