#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Dataloader utils

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
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

def read_file(_path):
    data = csv.DictReader(open(_path))
    aux = []
    id_list = []

    num_agent = 0
    for row in data:
        values = list(row.values())
        # object type 
        values[2] = 0 if values[2] == "AV" else 1 if values[2] == "AGENT" else 2
        if values[2] == 1:
            num_agent += 1
        # city
        values[-1] = 0 if values[-1] == "PIT" else 1
        # id
        id_list.append(values[1])
        # numpy_sequence
        aux.append(values)
 
    id_list, id_idx = np.unique(id_list, return_inverse=True)
    data = np.array(aux)
    data[:, 1] = id_idx

    return data.astype(np.float64)
    
def load_images(num_seq, obs_seq_data, first_obs, city_id, ego_origin, dist_rasterized_map, 
                object_class_id_list,debug_images=False):
    """
    Get the corresponding rasterized map
    """

    batch_size = len(object_class_id_list)
    frames_list = []

    # rasterized_start = time.time()
    t0_idx = 0
    for i in range(batch_size):
        
        curr_num_seq = int(num_seq[i].cpu().data.numpy())
        object_class_id = object_class_id_list[i].cpu().data.numpy()
         

        t1_idx = len(object_class_id_list[i]) + t0_idx
        if i < batch_size - 1:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:t1_idx,:]
        else:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:,:]
        curr_first_obs = first_obs[t0_idx:t1_idx,:]

        obs_len = curr_obs_seq_data.shape[0]

        curr_city = round(city_id[i])
        if curr_city == 0:
            city_name = "PIT"
        else:
            city_name = "MIA"

        curr_ego_origin = ego_origin[i].reshape(1,-1)
                                                     
        start = time.time()

        filename = data_imgs_folder + "/" + str(curr_num_seq) + ".png"

        img = map_utils.plot_trajectories(filename, curr_obs_seq_data, curr_first_obs, 
                                          curr_ego_origin, object_class_id, dist_rasterized_map,
                                          rot_angle=0,obs_len=obs_len, smoothen=True, show=False)

        end = time.time()
        # print(f"Time consumed by map generation and render: {end-start}")
        start = time.time()

        if debug_images:
            print("frames path: ", frames_path)
            print("curr seq: ", str(curr_num_seq))
            filename = frames_path + "seq_" + str(curr_num_seq) + ".png"
            print("path: ", filename)
            img = img * 255.0
            cv2.imwrite(filename,img)

        plt.close("all")
        end = time.time()
        frames_list.append(img)
        t0_idx = t1_idx

    # rasterized_end = time.time()
    # print(f"Time consumed by rasterized image: {rasterized_end-rasterized_start}")

    frames_arr = np.array(frames_list)
    return frames_arr

def load_goal_points(num_seq, obs_seq_data, first_obs, city_id, ego_origin, dist_rasterized_map, 
                    object_class_id_list,debug_images=False):
    """
    Get the corresponding rasterized map
    """

    batch_size = len(object_class_id_list)
    goal_points_list = []

    t0_idx = 0
    for i in range(batch_size):
        
        curr_num_seq = int(num_seq[i].cpu().data.numpy())
        object_class_id = object_class_id_list[i].cpu().data.numpy()
         
        t1_idx = len(object_class_id_list[i]) + t0_idx
        if i < batch_size - 1:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:t1_idx,:]
        else:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:,:]
        curr_first_obs = first_obs[t0_idx:t1_idx,:]

        obs_len = curr_obs_seq_data.shape[0]
        origin_pos = ego_origin[i][0]#.reshape(1,-1)
                                                     
        filename = data_imgs_folder + str(curr_num_seq) + ".png"
        
        agent_index = np.where(object_class_id == 1)[0].item()
        agent_obs_seq = curr_obs_seq_data[:,agent_index,:] # 20 x 2
        agent_first_obs = curr_first_obs[agent_index,:] # 1 x 2

        agent_obs_seq_abs = relative_to_abs(agent_obs_seq, agent_first_obs) # "abs" (around 0)
        agent_obs_seq_global = agent_obs_seq_abs + origin_pos # abs (hdmap coordinates)

        goal_points = dataset_utils.get_goal_points(filename, agent_obs_seq_global, origin_pos, dist_around)

        goal_points_list.append(goal_points)
        t0_idx = t1_idx

    goal_points_array = np.array(goal_points_list)

    return goal_points_array

