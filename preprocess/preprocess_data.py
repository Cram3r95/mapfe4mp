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
import time
import os
import git
import pdb

from prodict import Prodict

# DL & Math

import numpy as np

# Custom imports

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

avm = ArgoverseMap()

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset
from model.datasets.argoverse.dataset_utils import load_list_from_folder, get_origin_and_city, read_file

#######################################

config_path = os.path.join(BASE_DIR,"config/config_social_lstm_mhsa.yml")
with open(config_path) as config:
    config = yaml.safe_load(config)
    config = Prodict.from_dict(config)
    config.base_dir = BASE_DIR

config.dataset.start_from_percentage = 0.0

# Preprocess data
                         # Split, Process, Split percentage
splits_to_process = dict({"train":[True,0.1],
                          "val":[True,0.1],
                          "test":[False,1.0]})

PREPROCESS_SOCIAL_DATA = False
PREPROCESS_RELEVANT_CENTERLINES = True

for split_name,features in splits_to_process.items():
    if features[0]:
        # Agents trajectories

        if PREPROCESS_SOCIAL_DATA:
            ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                            root_folder=os.path.join(config.base_dir,config.dataset.path),
                                            obs_len=config.hyperparameters.obs_len,
                                            pred_len=config.hyperparameters.pred_len,
                                            distance_threshold=config.hyperparameters.distance_threshold,
                                            split=split_name,
                                            split_percentage=features[1],
                                            batch_size=config.dataset.batch_size,
                                            class_balance=config.dataset.class_balance,
                                            obs_origin=config.hyperparameters.obs_origin,
                                            preprocess_data=True,
                                            save_data=True)    

        # Most relevant lanes around the target agent

        if PREPROCESS_RELEVANT_CENTERLINES:
            folder = os.path.join(BASE_DIR,config.dataset.path,split_name,"data")
            files, num_files = load_list_from_folder(folder)
            afl = ArgoverseForecastingLoader(folder)

            file_id_list = []
            root_file_name = None
            for file_name in files:
                if not root_file_name:
                    root_file_name = os.path.dirname(os.path.abspath(file_name))
                file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
                file_id_list.append(file_id)
            file_id_list.sort()
            print("Num files: ", num_files)

            start_from = int(config.dataset.start_from_percentage*num_files)
            n_files = int(features[1]*num_files)

            if (start_from + n_files) >= num_files:
                file_id_list = file_id_list[start_from:]
            else:
                file_id_list = file_id_list[start_from:start_from+n_files]

            time_per_iteration = float(0)
            aux_time = float(0)
            candidate_centerlines_list = []

            for i, file_id in enumerate(file_id_list):
                print(f"File {file_id} -> {i+1}/{len(file_id_list)}")
                files_remaining = len(file_id_list) - (i+1)
                path = os.path.join(root_file_name,str(file_id)+".csv")

                start = time.time()

                data = read_file(path) 
                origin_pos, city_name = get_origin_and_city(data,config.hyperparameters.obs_origin)

                agent_obs_traj = afl.get(path).agent_traj[:config.hyperparameters.obs_len]
                candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, city_name, viz=False)
                print("Num relevant centerlines: ", len(candidate_centerlines))
                candidate_centerlines_list.append(candidate_centerlines)

                end = time.time()

                aux_time += (end-start)
                time_per_iteration = aux_time/(i+1)

                print(f"Time per iteration: {time_per_iteration} s. \n \
                        Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

            filename = os.path.join(BASE_DIR,config.dataset.path,split_name,
                                    f"data_processed_{str(int(features[1]*100))}_percent","relevant_centerlines.npy")

            with open(filename, 'wb') as my_file: np.save(my_file, candidate_centerlines_list)