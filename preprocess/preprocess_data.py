#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Preprocess the raw data. Get .npy files in order to avoid processing
## the trajectories and map information each time

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import yaml
from prodict import Prodict
import os
import git

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

# Custom imports

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset

#######################################

with open(r'../config/config_social_lstm_mhsa.yml') as config:
    config = yaml.safe_load(config)
    config = Prodict.from_dict(config)
    config.base_dir = BASE_DIR

# Preprocess data
                         # Split, Process, Split percentage
splits_to_process = dict({"train":[True,1.0],
                          "val":[False,1.0],
                          "test":[False,1.0]})

for k,v in splits_to_process.items():
    if v[0]:
        ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                          root_folder=os.path.join(config.base_dir,config.dataset.path),
                                          obs_len=config.hyperparameters.obs_len,
                                          pred_len=config.hyperparameters.pred_len,
                                          distance_threshold=config.hyperparameters.distance_threshold,
                                          split=k,
                                          split_percentage=v[1],
                                          batch_size=config.dataset.batch_size,
                                          class_balance=config.dataset.class_balance,
                                          obs_origin=config.hyperparameters.obs_origin,
                                          preprocess_data=True,
                                          save_data=True)    
