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

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

# Custom imports

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset

#######################################

with open(r'./config/config_social_lstm_mhsa.yml') as config:
    config = yaml.safe_load(config)
    config = Prodict.from_dict(config)
    config.base_dir = BASE_DIR

# Preprocess data

PREPROCESS_DATA_TRAIN = True
PREPROCESS_DATA_VAL = True

## Train split

if PREPROCESS_DATA_TRAIN:
    ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                      root_folder=config.dataset.path,
                                      obs_len=config.hyperparameters.obs_len,
                                      pred_len=config.hyperparameters.pred_len,
                                      distance_threshold=config.hyperparameters.distance_threshold,
                                      split="train",
                                      split_percentage=0.1,
                                      shuffle=config.dataset.shuffle,
                                      batch_size=config.dataset.batch_size,
                                      class_balance=config.dataset.class_balance,
                                      obs_origin=config.hyperparameters.obs_origin,
                                      preprocess_data=True,
                                      save_data=True)

## Val split

if PREPROCESS_DATA_VAL:
    ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                    root_folder=config.dataset.path,
                                    obs_len=config.hyperparameters.obs_len,
                                    pred_len=config.hyperparameters.pred_len,
                                    distance_threshold=config.hyperparameters.distance_threshold,
                                    split="val",
                                    split_percentage=0.1,
                                    shuffle=config.dataset.shuffle,
                                    batch_size=config.dataset.batch_size,
                                    class_balance=-1.0,
                                    obs_origin=config.hyperparameters.obs_origin,
                                    preprocess_data=True,
                                    save_data=True)