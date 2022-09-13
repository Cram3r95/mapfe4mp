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
import pandas as pd

from prodict import Prodict

# DL & Math

import math
import numpy as np

# Custom imports

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

avm = ArgoverseMap()

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from model.datasets.argoverse.map_functions import MapFeaturesUtils
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
modes_centerlines = ["train"] # if train -> compute the best candidate (oracle)
                              # if test, return N plausible candidates

PREPROCESS_SOCIAL_DATA = False
PREPROCESS_RELEVANT_CENTERLINES = True

obs_len = 20 # steps
pred_len = 30 # steps
freq = 10 # Hz ("steps/s")
obs_origin = 20 
min_vel = 25
max_points = 40

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

map_features_utils_instance = MapFeaturesUtils()
avm = ArgoverseMap()
viz = False

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

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
                seq_path = os.path.join(root_file_name,str(file_id)+".csv")

                start = time.time()

                # Compute centerlines 

                # data = read_file(seq_path) 
                # origin_pos, city_name = get_origin_and_city(data,config.hyperparameters.obs_origin)

                # agent_obs_traj = afl.get(seq_path).agent_traj[:config.hyperparameters.obs_len]
                # candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, city_name, viz=False)
                # print("Num relevant centerlines: ", len(candidate_centerlines))
                # candidate_centerlines_list.append(candidate_centerlines)

                # Compute map features using Argoverse Forecasting Baseline algorithm

                start = time.time()

                df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

                # Get social and map features for the agent

                agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
                city_name = agent_track[0,RAW_DATA_FORMAT["CITY_NAME"]]
                agent_xy = agent_track[:,[RAW_DATA_FORMAT["X"],RAW_DATA_FORMAT["Y"]]].astype("float")
                first_obs = agent_xy[0,:]
                origin_pos = agent_xy[obs_origin-1,:]

                for mode in modes_centerlines:
                    # Map features extraction

                    map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
                            agent_track,
                            file_id,
                            split_name,
                            obs_len,
                            obs_len + pred_len,
                            RAW_DATA_FORMAT,
                            mode,
                            avm,
                            viz
                        )

                    if mode == "train": # only best centerline ("oracle")
                        start_ = time.time()
                        oracle_centerline = map_feature_helpers["ORACLE_CENTERLINE"]
                        
                        first_obs_rep = np.repeat(first_obs.reshape(1,-1),repeats=oracle_centerline.shape[0],axis=0)

                        dist_array = np.zeros((first_obs_rep.shape[0]))
                        for k in range(first_obs_rep.shape[0]): # TODO: How to compute the L2-distance point by point?
                            dist_array[k] = np.linalg.norm(oracle_centerline[k,:]-first_obs_rep[k,:])
                        
                        lane_direction, yaw, confidence = avm.get_lane_direction_custom(origin_pos, city_name, visualize=viz)
                        # print("Degrees: ", math.degrees(yaw))

                        # TODO: The acceleration is quite noisy. Improve this calculation
                        vel, acc = avm.get_agent_velocity_and_acceleration(agent_xy[:obs_len,:])
                        if vel < min_vel:
                            vel = min_vel # Estimate a prediction horizon considering at least 25 m/s

                        # Reduce the lane to the closest fifty points, starting from the first observation
                        # and ending in the closest wp assuming CTRV during pred seconds

                        closest_wp = np.argmin(dist_array)
                        dist_array = dist_array[closest_wp:]

                        pred_horizon = pred_len/freq * vel
                        idx, value = find_nearest(dist_array,pred_horizon)
                        num_points = idx+1
                        print("Num points: ", num_points)

                        oracle_centerline_filtered = oracle_centerline[closest_wp:closest_wp+num_points,:]

                        if oracle_centerline_filtered.shape[0] != max_points:
                            interpolated_centerline = map_features_utils_instance.interpolate_centerline(split_name,
                                                                                            file_id,
                                                                                            oracle_centerline_filtered,
                                                                                            max_points=max_points,
                                                                                            viz=True)

                            candidate_centerlines_list.append(interpolated_centerline)

                        end_ = time.time()
                        print("Time consumed by L2 norm: ", end_-start_)

                end = time.time()

                aux_time += (end-start)
                time_per_iteration = aux_time/(i+1)

                print(f"Time per iteration: {time_per_iteration} s. \n \
                        Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

            filename = os.path.join(BASE_DIR,config.dataset.path,split_name,
                                    f"data_processed_{str(int(features[1]*100))}_percent","relevant_centerlines.npy")

            with open(filename, 'wb') as my_file: np.save(my_file, candidate_centerlines_list)

        