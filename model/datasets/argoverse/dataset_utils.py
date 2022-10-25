#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Dataloader utils

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import copy
import os
import csv
import glob, glob2
import pdb
import time

# DL & Math imports

import numpy as np
import torch
import cv2
import pandas as pd

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

import model.datasets.argoverse.goal_points_functions as goal_points_functions
import model.datasets.argoverse.map_functions as map_functions

#######################################

# Global variables

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

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

def apply_percentage_startfrom(file_id_list, num_files, 
                               shuffle=False,split_percentage=0.25, start_from_percentage=0.0):
    """
    """

    start_from = int(start_from_percentage*num_files)
    n_files = int(split_percentage*num_files)
    file_id_list = file_id_list[start_from:start_from+n_files]

    if (start_from + n_files) > num_files:
        print(f"WARNING: Unable to analyze {n_files} from file {start_from} \
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

    try:
        # Get [x,y] of the AGENT (object_class = 1) in the obs window
        origin = obs_frame[obs_frame[:,2] == 1][:,3:5].reshape(-1).tolist() 
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

def load_processed_files_from_npy(folder, required_variables_name_list):
    """
    """

    preprocessed_files, num_files = load_list_from_folder(folder)

    keys = []
    for preprocessed_file in preprocessed_files:
        key = preprocessed_file.split('/')[-1].split('.')[0]

        if key in required_variables_name_list:
            keys.append(key)

    try:
        assert len(keys) == len(required_variables_name_list), \
            "The number of loaded files from the folder does not match the number of variables you want"
    except:
        pdb.set_trace()

    preprocessed_data_dict = dict()
    for preprocessed_file in preprocessed_files:

        key = preprocessed_file.split('/')[-1].split('.')[0]
        
        if key in keys:
            with open(preprocessed_file, 'rb') as my_file:

                # Only load npy and npz files

                if (preprocessed_file.find('npy') != -1):
                    value = np.load(my_file)
                    preprocessed_data_dict[key] = value
                elif (preprocessed_file.find('npz') != -1):
                    value = np.load(my_file, allow_pickle=True)
                    value = value['arr_0'].item() # We need to extract the information at this point
                                                # Otherwise, if we store the NPz file in a dict and
                                                # we try to extract the information later, we get this 
                                                # error: "ValueError": seek of closed file
                    preprocessed_data_dict[key] = value
        
    return preprocessed_data_dict

# Physical information functions

def load_physical_information(num_seq_list, obs_traj, obs_traj_rel, pred_traj_gt, pred_traj_gt_rel,
                              first_obs, map_origin, dist_rasterized_map, object_class_id_list, 
                              data_imgs_folder, physical_context="dummies",relevant_centerlines=None,
                              DEBUG_IMAGES=False, DEBUG_TIME=False):
    """
    Get the physical context (rasterized map around the vehicle including trajectories), plausible
    goal points, or dummies
    """

    root_folder = os.path.join(*data_imgs_folder.split('/')[:-1])
    obs_len = obs_traj_rel.shape[0]

    batch_size = len(object_class_id_list)
    dist_around = abs(dist_rasterized_map[0])
    physical_context_list = []

    t0_idx = 0
    for i in range(batch_size):    
        curr_num_seq = int(num_seq_list[i].cpu().data.numpy())
        curr_object_class_id = object_class_id_list[i].cpu().data.numpy()
        agent_index = np.where(curr_object_class_id == 1)[0].item()

        t1_idx = len(object_class_id_list[i]) + t0_idx
        if i < batch_size - 1:
            curr_obs_traj = obs_traj[:,t0_idx:t1_idx,:]
            curr_obs_traj_rel = obs_traj_rel[:,t0_idx:t1_idx,:]
            curr_pred_traj_gt = pred_traj_gt[:,t0_idx:t1_idx,:]
            curr_pred_traj_gt_rel = pred_traj_gt_rel[:,t0_idx:t1_idx,:]
        else:
            curr_obs_traj = obs_traj[:,t0_idx:,:]
            curr_obs_traj_rel = obs_traj_rel[:,t0_idx:,:]
            curr_pred_traj_gt = pred_traj_gt[:,t0_idx:,:]
            curr_pred_traj_gt_rel = pred_traj_gt_rel[:,t0_idx:,:]

        curr_first_obs = first_obs[t0_idx:t1_idx,:]

        obs_len = curr_obs_traj_rel.shape[0]
        curr_map_origin = map_origin[i][0]#.reshape(1,-1)
                                                     
        filename = os.path.join(data_imgs_folder,str(curr_num_seq) + ".png")

        if physical_context == "visual":
            # TODO: Not used at this moment, but could be interesting. 
            # Use the new plot functions using Argoverse standard
            # The img should be normalized between 0 and 1
            img = map_functions.plot_trajectories(filename, curr_obs_traj_rel, curr_first_obs, 
                                                  curr_map_origin, curr_object_class_id, dist_rasterized_map,
                                                  rot_angle=-1,obs_len=obs_len, smoothen=True, show=False)
            if DEBUG_IMAGES:
                print("frames path: ", data_imgs_folder)
                print("curr seq: ", str(curr_num_seq))
                filename = data_imgs_folder + "seq_" + str(curr_num_seq) + ".png"
                print("path: ", filename)
                img = img * 255.0
                cv2.imwrite(filename,img)
            
            plt.close("all")
            physical_context_list.append(img)

        elif physical_context == "goals":
            agent_obs_seq = curr_obs_traj[:,agent_index,:] # 20 x 2, "abs" (around 0)
            agent_obs_seq_global = agent_obs_seq + curr_map_origin # abs (hdmap coordinates)

            goal_points = goal_points_functions.get_goal_points(filename, agent_obs_seq_global, curr_map_origin, dist_around)
            physical_context_list.append(goal_points)

        elif physical_context == "plausible_centerlines+area":
            filename = os.path.join(data_imgs_folder,str(curr_num_seq) + "_binary_plausible_area_filtered.png")
            curr_relevant_centerlines = relevant_centerlines[i]
            agent_obs_seq = curr_obs_traj[:,agent_index,:]
            agent_pred_gt_seq = curr_pred_traj_gt[:,agent_index,:]
            agent_xy_abs = np.vstack((agent_obs_seq,agent_pred_gt_seq))

            # Check if curr_map_origin is the same than last observation of current sequence

            debug_last_obs = False

            if debug_last_obs:
                csv_filename = os.path.join(root_folder,"data",f"{curr_num_seq}.csv")
                df = pd.read_csv(csv_filename, dtype={"TIMESTAMP": str})
                agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
                agent_xy = agent_track[:,[RAW_DATA_FORMAT["X"],RAW_DATA_FORMAT["Y"]]].astype("float")
                last_obs = agent_xy[obs_len-1,:]

                try:
                    assert np.allclose(curr_map_origin, last_obs), "Curr map origin and last observation do not match!"
                except:
                    pdb.set_trace()

            start = time.time()
            phy_info = goal_points_functions.get_driveable_area_and_centerlines(filename, agent_xy_abs, curr_relevant_centerlines, 
                                                                                curr_map_origin, IMG_ROWS=400, DEBUG=DEBUG_IMAGES,
                                                                                DEBUG_TIME=DEBUG_TIME)
            physical_context_list.append(phy_info)
            end = time.time()
            # print("Time consumed by goal function: ", end-start)

        t0_idx = t1_idx

    physical_context_arr = np.array(physical_context_list)
    return physical_context_arr

# Relative - Absolute coordinates

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
      N.B. If you only have the predictions, this must be the last observation.
           If you have the whole trajectory (obs+pred), this must be the first observation,
           since you must reconstruct the relative displacements from this position 
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2) (around 0,0, not map coordinates)
    """
    
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1) # Sum along the seq_len dimension!
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos

    return abs_traj.permute(1, 0, 2)

def relative_to_abs_multimodal(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch_size, num_modes, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch_size, 2)
      N.B. If you only have the predictions, this must be the last observation.
           If you have the whole trajectory (obs+pred), this must be the first observation,
           since you must reconstruct the relative displacements from this position 
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2) (around 0,0, not map coordinates)
    """

    displacement = torch.cumsum(rel_traj, dim=2) # Sum along the seq_len dimension!
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=1) # batch, 1 (only one position) x 1 (same for all modes) x 2
    abs_traj = displacement + start_pos

    return abs_traj
