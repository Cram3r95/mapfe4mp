#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Preprocess the raw data. Get .npy files in order to avoid processing
## the trajectories and map information each time

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import random
import math
import pdb
import copy
import sys
import yaml
from prodict import Prodict

# DL & Math imports

import numpy as np
import torch
import cv2
from sklearn import linear_model

# Plot imports

import matplotlib.pyplot as plt

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

# Custom imports

from model.data_loader.argoverse.dataloader.dataloader import ArgoverseMotionForecastingDataset, seq_collate

#######################################

with open(r'./configs/mp_so_goals.yml') as config:
    config = yaml.safe_load(config)
    config = Prodict.from_dict(config)
    config.base_dir = BASE_DIR

# Generate data

ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                  root_folder=config.dataset.path,
                                  obs_len=config.hyperparameters.obs_len,
                                  pred_len=config.hyperparameters.pred_len,
                                  distance_threshold=config.hyperparameters.distance_threshold,
                                  split="train",
                                  num_agents_per_obs=config.hyperparameters.num_agents_per_obs,
                                  split_percentage=config.dataset.split_percentage,
                                  shuffle=config.dataset.shuffle,
                                  batch_size=config.dataset.batch_size,
                                  class_balance=config.dataset.class_balance,
                                  obs_origin=config.hyperparameters.obs_origin)