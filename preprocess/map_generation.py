#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Offline map generation

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import yaml
import numpy as np
from prodict import Prodict
import pdb
import sys
import os
import time

# DL & Math imports

import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

from model.datasets.argoverse.dataset_utils import read_file, \
                                                   load_list_from_folder
                                                   
import model.datasets.argoverse.map_functions as map_functions
from argoverse.map_representation.map_api import ArgoverseMap

#######################################

avm = ArgoverseMap()

def get_origin_and_city(seq,obs_window):
    """
    """

    frames = np.unique(data[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

    obs_frame = frame_data[obs_window-1]

    # Get origin

    try:
        origin = obs_frame[obs_frame[:,2] == 1][:,3:5].reshape(-1) # Get x|y of the AGENT (object_class = 1) in the obs window
    except:
        pdb.set_trace()
    
    # Get city

    city_id = round(obs_frame[0,-1])
    if city_id == 0:
        city_name = "PIT"
    else:
        city_name = "MIA"

    return origin, city_name

# Load config

config_path = BASE_DIR + '/config/config_social_lstm_mhsa.yml'

with open(config_path) as config:
        config = yaml.safe_load(config)
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR

# Fill some additional dimensions

past_observations = config.hyperparameters.obs_len

config.dataset.split = "val"
config.dataset.split_percentage = 1.0 # To generate the final results, must be 1 (whole split test)
config.dataset.start_from_percentage = 0.0
config.dataset.batch_size = 1 
config.dataset.num_workers = 0
config.dataset.class_balance = -1.0 # Do not consider this to preprocess the map
config.dataset.shuffle = False
config.dataset.data_augmentation = False

config.hyperparameters.pred_len = 30 # In test, we do not have the gt (prediction points)

data_images_folder = BASE_DIR + "/" + config.dataset.path + config.dataset.split + "/data_images_200mx200m"

MAP_GENERATION = True

dist_around = 100
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

if MAP_GENERATION:
    # Only load the city and x|y center to generate the background

    obs_window = 20

    folder = BASE_DIR + "/" + config.dataset.path + config.dataset.split + "/data/"
    files, num_files = load_list_from_folder(folder)

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
    n_files = int(config.dataset.split_percentage*num_files)

    if (start_from + n_files) >= num_files:
        file_id_list = file_id_list[start_from:]
    else:
        file_id_list = file_id_list[start_from:start_from+n_files]

    time_per_iteration = float(0)
    aux_time = float(0)

    for i, file_id in enumerate(file_id_list):
        print(f"File {file_id} -> {i+1}/{len(file_id_list)}")
        files_remaining = len(file_id_list) - (i+1)
        path = os.path.join(root_file_name,str(file_id)+".csv")
        data = read_file(path) 

        origin_pos, city_name = get_origin_and_city(data,config.hyperparameters.obs_origin)
        origin_pos = origin_pos.tolist() # np.array -> list

        start = time.time()
        map_functions.map_generator(file_id,
                                    origin_pos,
                                    dist_rasterized_map,
                                    avm,
                                    city_name,
                                    show=False,
                                    root_folder=data_images_folder)
        end = time.time()
        aux_time += (end-start)
        time_per_iteration = aux_time/(i+1)

        print(f"Time per iteration: {time_per_iteration} s. \n \
                Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

        plt.close("all")