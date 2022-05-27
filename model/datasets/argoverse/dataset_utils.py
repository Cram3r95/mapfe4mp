#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Dataloader utils

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import copy
import os
import csv
import time
import glob, glob2
import pdb
import random

# DL & Math imports

import numpy as np
import torch
import cv2

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

import model.datasets.argoverse.goal_points_functions as goal_points_functions
import model.datasets.argoverse.map_functions as map_functions

#######################################

# File functions

def isstring(string_test):
    """
    """
    return isinstance(string_test, str)

def safe_path(input_path):
    """
    """
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None):
    """
    """
    folder_path = safe_path(folder_path)
    if isstring(ext_filter): ext_filter = [ext_filter]

    full_list = []
    if depth is None: # Find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = os.path.join(wildcard_prefix,'*'+ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path,wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            full_list += curlist
    else: # Find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            full_list += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            full_list += newlist

    full_list = [os.path.normpath(path_tmp) for path_tmp in full_list]
    num_elem = len(full_list)

    return full_list, num_elem

def get_sorted_file_id_list(files):
    """
    """

    sorted_file_id_list = []
    root_file_name = None
    for file_name in files:
        if not root_file_name:
            root_file_name = os.path.dirname(os.path.abspath(file_name))
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        sorted_file_id_list.append(file_id)
    sorted_file_id_list.sort()

    return sorted_file_id_list, root_file_name

def apply_shuffling_percentage_startfrom(file_id_list, num_files, 
                                         shuffle=False, split_percentage=0.25, start_from_percentage=0.0):
    """
    """

    if shuffle:
        rng = np.random.default_rng()
        indeces = rng.choice(num_files, size=int(num_files*split_percentage), replace=False)
        file_id_list = np.take(file_id_list, indeces, axis=0)
    else:
        start_from = int(start_from_percentage*num_files)
        n_files = int(split_percentage*num_files)
        file_id_list = file_id_list[start_from:start_from+n_files]

        if (start_from + n_files) >= num_files:
            print(f"WARNING: Unable to analyze {n_files} from {start_from} file. \
                    Analyzing remaining files to completion")
            file_id_list = file_id_list[start_from:]
        else:
            file_id_list = file_id_list[start_from:start_from+n_files]
    print("Num files to be analized: ", len(file_id_list)) 

    return file_id_list

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

def get_origin_and_city(seq,obs_window):
    """
    """

    frames = np.unique(seq[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(seq[frame == seq[:, 0], :]) # save info for each frame

    obs_frame = frame_data[obs_window-1]
    # pdb.set_trace()
    try:
        origin = obs_frame[obs_frame[:,2] == 1][:,3:5] # Get x|y of the AGENT (object_class = 1) in the obs window
    except:
        pdb.set_trace()
    city_id = round(obs_frame[0,-1])
    if city_id == 0:
        city_name = "PIT"
    else:
        city_name = "MIA"
    
    return origin, city_name

def create_dictionary_from_variable_list(variable_list, variable_name_list):
    """
    """

    preprocessed_data_dict = dict()
    #pdb.set_trace()
    for key,value in zip(variable_name_list,variable_list):
        preprocessed_data_dict[key] = value
    return preprocessed_data_dict

def save_processed_data_as_npy(data_processed_folder, 
                               processed_data_dict,
                               split_percentage):
    """
    """

    if not os.path.exists(data_processed_folder):
        print("Create path: ", data_processed_folder)
        os.mkdir(data_processed_folder)

    for key, value in processed_data_dict.items():
        filename = data_processed_folder + "/" + key + ".npy"
        with open(filename, 'wb') as my_file: np.save(my_file, value)
    
    split = data_processed_folder.split('/')[-2]

    # Save text file with additional information

    filename = data_processed_folder + "/readme.txt"
    split_percentage *= 100
    string = f'This folder contains {split_percentage} % of the original files (after being processed) of the {split} split'.encode()
    with open(filename, 'wb') as my_file:
        my_file.write(string)

def load_processed_files_from_npy(folder):
    """
    """

    preprocessed_data_dict = dict()

    preprocessed_files, num_files = load_list_from_folder(folder)
    for preprocessed_file in preprocessed_files:
        key = preprocessed_file.split('/')[-1].split('.')[0]
        with open(preprocessed_file, 'rb') as my_file:
            if (preprocessed_file.find('npy') != -1): # Do not load other files
                value = np.load(my_file)
        preprocessed_data_dict[key] = value

    return preprocessed_data_dict

def load_images(num_seq, obs_seq_data, first_obs, city_id, ego_origin, dist_rasterized_map, 
                object_class_id_list,data_imgs_folder,debug_images=False):
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

        img = map_functions.plot_trajectories(filename, curr_obs_seq_data, curr_first_obs, 
                                          curr_ego_origin, object_class_id, dist_rasterized_map,
                                          rot_angle=0,obs_len=obs_len, smoothen=True, show=False)

        end = time.time()
        # print(f"Time consumed by map generation and render: {end-start}")
        start = time.time()

        if debug_images:
            print("frames path: ", data_imgs_folder)
            print("curr seq: ", str(curr_num_seq))
            filename = data_imgs_folder + "seq_" + str(curr_num_seq) + ".png"
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

def load_goal_points(num_seq_list, obs_traj_rel, first_obs, city_id, ego_vehicle_origin, 
                     dist_rasterized_map, object_class_id_list, data_imgs_folder,debug_images=False):
    """
    Get plausible goal points
    """

    batch_size = len(object_class_id_list)
    dist_around = abs(dist_rasterized_map[0])
    goal_points_list = []

    t0_idx = 0
    for i in range(batch_size):    
        curr_num_seq = int(num_seq_list[i].cpu().data.numpy())
        object_class_id = object_class_id_list[i].cpu().data.numpy()
         
        t1_idx = len(object_class_id_list[i]) + t0_idx
        if i < batch_size - 1:
            curr_obs_seq_data = obs_traj_rel[:,t0_idx:t1_idx,:]
        else:
            curr_obs_seq_data = obs_traj_rel[:,t0_idx:,:]
        curr_first_obs = first_obs[t0_idx:t1_idx,:]

        obs_len = curr_obs_seq_data.shape[0]
        origin_pos = ego_vehicle_origin[i][0]#.reshape(1,-1)
                                                     
        filename = data_imgs_folder + str(curr_num_seq) + ".png"
        print("filename: ", filename)
        
        agent_index = np.where(object_class_id == 1)[0].item()
        agent_obs_seq = curr_obs_seq_data[:,agent_index,:] # 20 x 2
        agent_first_obs = curr_first_obs[agent_index,:] # 1 x 2

        agent_obs_seq_abs = relative_to_abs(agent_obs_seq, agent_first_obs) # "abs" (around 0)
        agent_obs_seq_global = agent_obs_seq_abs + origin_pos # abs (hdmap coordinates)

        goal_points = goal_points_functions.get_goal_points(filename, agent_obs_seq_global, origin_pos, dist_around)

        goal_points_list.append(goal_points)
        t0_idx = t1_idx

    goal_points_array = np.array(goal_points_list)

    return goal_points_array

# Relative - Absolute coordinates

def relative_to_abs_sgan(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def relative_to_abs_sgan_multimodal(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (b, m, t, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    displacement = torch.cumsum(rel_traj, dim=2)
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=1)
    abs_traj = displacement + start_pos
    return abs_traj

def relative_to_abs(rel_traj, start_pos):
    # TODO: Not used in the training stage. Nevertheless, rewrite using torch, not numpy
    """
    Inputs:
    - rel_traj: numpy array of shape (len, 2)
    - start_pos: numpy array of shape (1, 2)
    Outputs:
    - abs_traj: numpy array of shape (len, 2) in absolute coordinates
    """

    displacement = np.cumsum(rel_traj, axis=0)
    abs_traj = displacement + start_pos

    return abs_traj
