#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Argoverse dataset

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import random
import math
import os
import time
import operator
import copy
import pdb

# DL & Math imports

import numpy as np
import torch
from torch.utils.data import Dataset

# Custom imports

import model.datasets.argoverse.dataset_utils as dataset_utils
import model.datasets.argoverse.geometric_functions as geometric_functions
import model.datasets.argoverse.data_augmentation_functions as data_augmentation_functions
import model.datasets.argoverse.plot_functions as plot_functions

#######################################

# Data augmentation variables

APPLY_DATA_AUGMENTATION = False
DEBUG_DATA_AUGMENTATION = False

decision = [0,1] # Not apply/apply
dropout_prob = [0.5,0.5] # Not applied/applied probability
gaussian_noise_prob = [0.5,0.5]
rotation_prob = [0.5,0.5]

mu,std = 0,0.5
rotation_angles = [90,180,270]
rotation_angles_prob = [0.33,0.33,0.34]

# Auxiliar variables

data_imgs_folder = None
PHYSICAL_CONTEXT = "Dummies"

dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

#######################################

# Main dataset functions

def seq_collate(data):
    """
    This function takes the output of __getitem__ and returns a specific format
    that will be used by the PyTorch class to output the data (used by the model)

    N.B. Data augmentation must be always in the seq_collate function in order
    to always generate different data, not in the main class where the preprocessed
    data is computed.
    """

    start = time.time()

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_obj, loss_mask, seq_id_list, object_class_id_list, 
     object_id_list, city_id, map_origin, num_seq_list, norm) = zip(*data)

    _len = [len(seq) for seq in obs_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_traj, dim=0).permute(2, 0, 1) # Past Observations x Num_agents · batch_size x 2                                                          
    pred_traj_gt = torch.cat(pred_traj_gt, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=0).permute(2, 0, 1)
    pred_traj_gt_rel = torch.cat(pred_traj_gt_rel, dim=0).permute(2, 0, 1)
    non_linear_obj = torch.cat(non_linear_obj)
    loss_mask = torch.cat(loss_mask, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end) # This variable represents the number of agents per 
                                                    # sequence in the batch, i.e. if this variable = [[0,3],[4,10]],
                                                    # that means that obs_traj = 20 x 10 x 2, with batch_size = 2, and
                                                    # there are 3 agents in the first element of the batch and 7 agents in 
                                                    # the second element of the batch
    batch_size = seq_start_end.shape[0]
    obs_len = obs_traj.shape[0]
    pred_len = pred_traj_gt.shape[0]
    id_frame = torch.cat(seq_id_list, dim=0).permute(2, 0, 1) # seq_len - objs_in_curr_seq - 3

    # Data augmentation

    curr_split = data_imgs_folder.split('/')[-3]

    APPLY_DATA_AUGMENTATION = True

    if APPLY_DATA_AUGMENTATION and curr_split == "train":
        aug_obs_traj = torch.zeros((obs_traj.shape))
        aug_obs_traj_rel = torch.zeros((obs_traj_rel.shape))
        aug_pred_traj_gt = torch.zeros((pred_traj_gt.shape))
        aug_pred_traj_gt_rel = torch.zeros((pred_traj_gt_rel.shape))

        i = 0
        for start,end in seq_start_end.data:
            num_obstacles = (end-start).item() # int
            curr_object_class_id_list = object_class_id_list[i]
            curr_origin = map_origin[i][0] # TODO: Save again the .npy data with 1D tensor [], not 2D tensor [[]] for each element
            seq_id = num_seq_list[i].item()

            # Original trajectories (observations and predictions)

            curr_obs_traj = obs_traj[:,start:end,:] 
            curr_first_obs = curr_obs_traj[0,:,:]
            curr_obs_traj_rel = obs_traj_rel[:,start:end,:]
            curr_pred_traj_gt = pred_traj_gt[:,start:end,:]
            curr_pred_traj_gt_rel = pred_traj_gt_rel[:,start:end,:]

            # Apply data augmentation for every sequence (scenario) of the batch

            aug_curr_obs_traj = copy.deepcopy(curr_obs_traj)
            aug_curr_pred_traj_gt = copy.deepcopy(curr_pred_traj_gt)

            apply_dropout = np.random.choice(decision,num_obstacles,p=dropout_prob) # For each obstacle of the sequence
            apply_gaussian_noise = np.random.choice(decision,num_obstacles,p=gaussian_noise_prob) # For each obstacle of the sequence
            apply_rotation = np.random.choice(decision,1,p=rotation_prob) # To the whole sequence
     
            if np.any(apply_dropout): # Not apply if all elements are 0
                aug_curr_obs_traj = data_augmentation_functions.dropout_points(aug_curr_obs_traj,
                                                                               apply_dropout,
                                                                               num_obs=obs_len,
                                                                               percentage=0.3)
            if np.any(apply_gaussian_noise): # Not apply if all elements are 0
                aug_curr_obs_traj = data_augmentation_functions.add_gaussian_noise(aug_curr_obs_traj,
                                                                                   apply_gaussian_noise,
                                                                                   num_obstacles,
                                                                                   num_obs=obs_len,
                                                                                   mu=0,sigma=0.5)

            ## N.B. If you apply rotation as data augmentation, the groundtruth must be rotated too!
            
            rot_angle = 0
            if np.any(apply_rotation): 
                rot_angle = np.random.choice(rotation_angles,1,p=rotation_angles_prob).item()

                aug_curr_obs_traj = data_augmentation_functions.rotate_traj(aug_curr_obs_traj,rot_angle)
                aug_curr_pred_traj_gt = data_augmentation_functions.rotate_traj(aug_curr_pred_traj_gt,rot_angle)

            ## Get first obs and new relatives after data augmentation for this sequence

            aug_curr_first_obs = aug_curr_obs_traj[0,:,:] 
            aug_curr_obs_traj_rel = torch.zeros((aug_curr_obs_traj.shape))
            aug_curr_obs_traj_rel[1:,:,:] = torch.sub(aug_curr_obs_traj[1:,:,:],
                                                      aug_curr_obs_traj[:-1,:,:])

            #rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1] # Get displacements between consecutive steps

            aug_curr_pred_traj_gt_rel = torch.zeros((aug_curr_pred_traj_gt.shape))
            aug_curr_pred_traj_gt_rel[1:,:,:] = torch.sub(aug_curr_pred_traj_gt[1:,:,:],
                                                          aug_curr_pred_traj_gt[:-1,:,:])

            ## Fill torch tensor (whole batch)

            aug_obs_traj[:,start:end,:] = aug_curr_obs_traj
            aug_obs_traj_rel[:,start:end,:] = aug_curr_obs_traj_rel
            aug_pred_traj_gt[:,start:end,:] = aug_curr_pred_traj_gt
            aug_pred_traj_gt_rel[:,start:end,:] = aug_curr_pred_traj_gt_rel

            if DEBUG_DATA_AUGMENTATION:
                filename = f"data/datasets/argoverse/motion-forecasting/train/data_images/{seq_id}.png"

                # Original observations (No data augmentation)

                curr_traj_rel = torch.cat((curr_obs_traj_rel,
                                           curr_pred_traj_gt_rel),dim=0)

                plot_functions.plot_trajectories(filename,curr_traj_rel,curr_first_obs,
                                                 curr_origin,curr_object_class_id_list,dist_rasterized_map,
                                                 rot_angle=-1,obs_len=obs_len,
                                                 smoothen=False, save=True)

                # New observations (after data augmentation)

                aug_curr_traj_rel = torch.cat((aug_curr_obs_traj_rel,
                                               aug_curr_pred_traj_gt_rel),dim=0)

                plot_functions.plot_trajectories(filename,aug_curr_traj_rel,aug_curr_first_obs,
                                                 curr_origin,curr_object_class_id_list,dist_rasterized_map,
                                                 rot_angle=rot_angle,obs_len=obs_len,
                                                 smoothen=False, save=True, data_aug=True)
            i += 1

            # Replace tensors

        obs_traj = aug_obs_traj
        obs_traj_rel = aug_obs_traj_rel
        pred_traj_gt = aug_pred_traj_gt
        pred_traj_gt_rel = aug_pred_traj_gt_rel

    # Get physical information (image or goal points. Otherwise, use dummies)

    # pdb.set_trace()

    start = time.time()

    first_obs = obs_traj[0,:,:] # 1 x agents · batch_size x 2

    if (PHYSICAL_CONTEXT == "visual"  # batch_size x channels x height x width 
     or PHYSICAL_CONTEXT == "goals"): # batch_size x num_goal_points x 2 (x|y) (real-world coordinates (HDmap))
        frames = dataset_utils.load_physical_information(num_seq_list, obs_traj_rel, first_obs, map_origin,
                                                         dist_rasterized_map, object_class_id_list, data_imgs_folder,
                                                         physical_context=PHYSICAL_CONTEXT,debug_images=False)
        frames = torch.from_numpy(frames).type(torch.float32)
        if PHYSICAL_CONTEXT == "visual": frames = frames.permute(0, 3, 1, 2)
    else: # dummy frames
        frames = np.random.randn(1,1,1,1)
        frames = torch.from_numpy(frames).type(torch.float32)

    end = time.time()
    # print(f"Time consumed by load_images function: {end-start}\n")

    object_cls = torch.cat(object_class_id_list, dim=0)
    obj_id = torch.cat(object_id_list, dim=0)
    map_origin = torch.stack(map_origin)
    num_seq_list = torch.stack(num_seq_list)
    norm = torch.stack(norm)

    out = [obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
           loss_mask, seq_start_end, frames, object_cls, obj_id, map_origin, num_seq_list, norm]

    end = time.time()
    # print(f"Time consumed by seq_collate function: {end-start}\n")

    return tuple(out)

def process_window_sequence(idx, frame_data, frames, obs_len, 
                            pred_len, file_id, split, obs_origin):
    """
    Input:
        idx (int): AV id
        frame_data array (n, 6):
            - timestamp (int)
            - id (int) -> previously need to be converted. Original data is string
            - type (int) -> need to be converted from string to int
            - x (float) -> x position (global, hdmap)
            - y (float) -> y position (global, hdmap)
            - city_name (int)
        obs_len (int)
        pred_len (int)
        threshold (float)
        file_id (int)
        split (str: "train", "val", "test") 
    Output:
        num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, curr_seq_rel, 
        id_frame_list, object_class_list, city_id, map_origin
    """

    seq_len = obs_len + pred_len

    # Prepare current sequence and get unique obstacles

    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
    objs_in_curr_seq = np.unique(curr_seq_data[:, 1]) # Unique IDs in the sequence

    # Initialize variables

    if split == "train" or split == "val": tensors_len = seq_len
    elif split == "test": tensors_len = obs_len

    curr_seq_rel = np.zeros((len(objs_in_curr_seq), 2, tensors_len)) # objs_in_curr_seq x 2 (x,y) x tensors_len (ej: 50)                              
    curr_seq = np.zeros((len(objs_in_curr_seq), 2, tensors_len)) # objs_in_curr_seq x 2 (x,y) x tensors_len (ej: 50)
    curr_loss_mask = np.zeros((len(objs_in_curr_seq), tensors_len)) # objs_in_curr_seq x tensors_len (ej: 50)
    object_class_list = np.zeros(len(objs_in_curr_seq)) 
    id_frame_list  = np.zeros((len(objs_in_curr_seq), 3, tensors_len))

    num_objs_considered = 0
    _non_linear_obj = []
    city_id = curr_seq_data[0,5]

    # Get map origin of this sequence. We assume we are going to take the AGENT as reference (object of interest in 
    # Argoverse 1.0). In this repository it is written ego_vehicle but actually it is NOT the ego-vehicle (AV, 
    # the vehicle which captures the scene), but another object of interest to be predicted. 
    # TODO: Change ego_vehicle_origin notation to just map_origin

    aux_seq = curr_seq_data[curr_seq_data[:, 2] == 1, :] # curr_seq_data[:, 2] represents the type. 1 == AGENT
    map_origin = aux_seq[obs_origin-1, 3:5] # x,y 

    # Iterate over all unique objects

    for _, obj_id in enumerate(objs_in_curr_seq):
        curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]

        # If the object has less than "seq_len" observations, discard.
        # If we are processing the "test" set, we only have the observations, not the predictions
                                                                                 
        pad_front = frames.index(curr_obj_seq[0, 0]) - idx
        pad_end = frames.index(curr_obj_seq[-1, 0]) - idx + 1

        # print("ID: ", curr_obj_seq[0,2])
        # print("Front End: ", pad_front, pad_end)
        if ((pad_end - pad_front != tensors_len) or (curr_obj_seq.shape[0] != tensors_len)):
            continue

        object_class_list[num_objs_considered] = curr_obj_seq[0,2] # 0 == AV, 1 == AGENT, 2 == OTHER

        # Record timestamp, object ID and file_id information (for each object)
        # id_frame_list represents a single sequence, so the second dimension indicates the object ID
        # in that sequence

        cache_tmp = np.transpose(curr_obj_seq[:,:2])
        id_frame_list[num_objs_considered, :2, :] = cache_tmp
        id_frame_list[num_objs_considered,  2, :] = file_id

        # Get x-y data (w.r.t the map origin, so they are absolute 
        # coordinates but in the local frame, not map (global) frame)
        
        curr_obj_seq = np.transpose(curr_obj_seq[:, 3:5])
        curr_obj_seq = curr_obj_seq - map_origin.reshape(-1,1)

        # Make coordinates relative (relative here means displacements between consecutive steps)

        rel_curr_obj_seq = np.zeros(curr_obj_seq.shape) 
        rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1] # Get displacements between consecutive steps
        
        _idx = num_objs_considered
        curr_seq[_idx, :, pad_front:pad_end] = curr_obj_seq
        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_obj_seq

        # Linear vs Non-Linear Trajectory

        if split != 'test':
            try:
                non_linear = geometric_functions.get_non_linear(file_id, curr_seq, idx=_idx, obj_kind=curr_obj_seq[0,2],
                                                                threshold=2, debug_trajectory_classifier=False)
            except: # E.g. All max_trials iterations were skipped because each randomly chosen sub-sample 
                    # failed the passing criteria. Return non-linear because RANSAC could not fit a model
                non_linear = 1.0
            _non_linear_obj.append(non_linear)
        curr_loss_mask[_idx, pad_front:pad_end] = 1

        # Add num_objs_considered

        num_objs_considered += 1

    return num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, curr_seq_rel, \
           id_frame_list, object_class_list, city_id, map_origin
           
class ArgoverseMotionForecastingDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, dataset_name, root_folder, obs_len=20, pred_len=30, distance_threshold=30,
                 split='train', split_percentage=0.1, start_from_percentage=0.0, 
                 batch_size=16, class_balance=-1.0, obs_origin=1, data_augmentation=False, 
                 physical_context="dummy", extra_data_train=-1.0, preprocess_data=False, save_data=False):
        super(ArgoverseMotionForecastingDataset, self).__init__()

        # Initialize class variables

        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.distance_threshold = distance_threshold # Monitorize distance_threshold around the AGENT
        self.split = split
        self.batch_size = batch_size
        self.class_balance = class_balance
        self.obs_origin = obs_origin
        self.min_objs = 2 # Minimum number of objects to include the scene (AV and AGENT)
        self.cont_seqs = 0
        self.cont_curved_trajs = 0
        self.data_augmentation = data_augmentation
        self.physical_context = physical_context
        self.extra_data_train = extra_data_train

        self.dataset_name = dataset_name
        self.root_folder = root_folder
        self.data_processed_folder = os.path.join(root_folder,
                                                  self.split,
                                                  f"data_processed_{str(int(split_percentage*100))}_percent")
                                                       
        if self.extra_data_train != -1.0:
            self.extra_data_processed_folder = os.path.join(root_folder,
                                                            "val",
                                                            f"data_processed_{str(int(split_percentage*100))}_percent")
            self.class_balance = -1.0 # TODO: If we merge data from validation and train, then the stored variables
                                        # to do class balance are useless. Check this
                                        #  
            if self.split == "val":
                self.extra_data_train = 1 - self.extra_data_train # Use remaining files for validation

        variable_name_list = ['seq_list','seq_list_rel','loss_mask_list','non_linear_obj',
                              'num_objs_in_seq','seq_id_list','object_class_id_list',
                              'object_id_list','ego_vehicle_origin','num_seq_list',
                              'straight_trajectories_list','curved_trajectories_list','city_id','norm'] 
        
        # Preprocess data (from raw .csvs to torch Tensors, at least including the 
        # AGENT (most important vehicle) and AV

        assert os.path.isdir(os.path.join(root_folder,self.split)), "\n\nHey, the folder you want to analyze does not exist!"
             
        if (preprocess_data and not os.path.isdir(self.data_processed_folder)): # This split percentage has not been processed yet
                                                                                # (in order to avoid unwanted overwriting, just in case ...)
            folder = os.path.join(root_folder,self.split,"data")
            files, num_files = dataset_utils.load_list_from_folder(folder)

            # Sort list and analize a specific percentage/from a specific position

            file_id_list, root_file_name = dataset_utils.get_sorted_file_id_list(files)     
            file_id_list = dataset_utils.apply_percentage_startfrom(file_id_list, num_files, 
                                                                    split_percentage=split_percentage, 
                                                                    start_from_percentage=start_from_percentage)

            seq_list = [] # Absolute coordinates (obs+pred) around 0.0 (center of the local map)
            seq_list_rel = [] # Relative displacements (obs+pred)
            num_objs_in_seq = [] 
            loss_mask_list = []
            non_linear_obj = [] # Object with non-linear trajectory
            seq_id_list = []
            object_class_id_list = [] # 0 = AV, 1 = AGENT, 2 = DUMMY 
            object_id_list = [] 
            num_seq_list = [] # ID of the current sequence
            straight_trajectories_list = []
            curved_trajectories_list = []
            ego_vehicle_origin = [] # Origin of the AGENT (TODO: ego_vehicle_origin is a WRONG nomenclature)
            city_ids = []

            print("Start Dataset")
            # TODO: Speed-up dataloading, avoiding objects further than X distance

            time_per_iteration = float(0)
            aux_time = float(0)

            t0 = time.time()
            for i, file_id in enumerate(file_id_list):
                start = time.time()

                print(f"File {file_id} -> {i+1}/{len(file_id_list)}")
                files_remaining = len(file_id_list) - (i+1)

                num_seq_list.append(file_id)
                path = os.path.join(root_file_name,str(file_id)+".csv")
                data = dataset_utils.read_file(path) 
            
                frames = np.unique(data[:, 0]).tolist() # Get unique timestamps (50 in this case)
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

                idx = 0

                num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
                curr_seq_rel, id_frame_list, object_class_list, city_id, ego_origin = \
                    process_window_sequence(idx, frame_data, frames,
                                            self.obs_len, self.pred_len, 
                                            file_id, self.split, self.obs_origin)

                # Check if the current AGENT trajectory can be considered as a curve or straight trajectory
                # (so for further training we can focus on the most difficult samples -> sequences in which
                # the AGENT is performing a curved trajectory)
 
                if self.class_balance >= 0.0:
                    agent_idx = int(np.where(object_class_list==1)[0])

                    try:
                        non_linear = geometric_functions.get_non_linear(file_id, curr_seq, idx=agent_idx, obj_kind=1,
                                                                  threshold=2, debug_trajectory_classifier=False)
                    except: # E.g. All max_trials iterations were skipped because each randomly chosen sub-sample 
                            # failed the passing criteria. Return non-linear because RANSAC could not fit a model
                        non_linear = 1.0

                if num_objs_considered >= self.min_objs:
                    non_linear_obj += _non_linear_obj
                    seq_list.append(curr_seq[:num_objs_considered]) # Remove dummies
                    seq_list_rel.append(curr_seq_rel[:num_objs_considered])
                    num_objs_in_seq.append(num_objs_considered)
                    loss_mask_list.append(curr_loss_mask[:num_objs_considered])
                    ###################################################################
                    seq_id_list.append(id_frame_list[:num_objs_considered]) # (timestamp, id, file_id)
                    object_class_id_list.append(object_class_list[:num_objs_considered]) # obj_class (-1 0 1 2 2 2 2 ...)
                    object_id_list.append(id_frame_list[:num_objs_considered,1,0])
                    ###################################################################
                    city_ids.append(city_id)
                    ego_vehicle_origin.append(ego_origin)
                    ###################################################################
                    if self.class_balance >= 0.0:
                        if non_linear == 1.0:
                            curved_trajectories_list.append(file_id)
                        else:
                            straight_trajectories_list.append(file_id)

                end = time.time()
                aux_time += (end-start)
                time_per_iteration = aux_time/(i+1)


                print(f"Time per iteration: {time_per_iteration} s. \n \
                        Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

            print("Dataset time: ", time.time() - t0)

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0) # Objects x 2 x seq_len
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_obj = np.asarray(non_linear_obj)
            seq_id_list = np.concatenate(seq_id_list, axis=0)
            object_class_id_list = np.concatenate(object_class_id_list, axis=0)
            object_id_list = np.concatenate(object_id_list)
            num_seq_list = np.concatenate([num_seq_list])
            curved_trajectories_list = np.concatenate([curved_trajectories_list])
            straight_trajectories_list = np.concatenate([straight_trajectories_list])
            ego_vehicle_origin = np.asarray(ego_vehicle_origin)
            city_ids = np.asarray(city_ids)

            # Normalize abs and relative data ((your_vale - min) / (max - min))

            abs_norm = (seq_list.min(), seq_list.max())
            rel_norm = (seq_list_rel.min(), seq_list_rel.max())

            norm = (abs_norm, rel_norm)

            # Create dictionary with all processed data

            variable_list = [seq_list, seq_list_rel, loss_mask_list, non_linear_obj, num_objs_in_seq,
                             seq_id_list, object_class_id_list, object_id_list, ego_vehicle_origin, 
                             num_seq_list, straight_trajectories_list, curved_trajectories_list, city_ids, norm]
            preprocess_data_dict = dataset_utils.create_dictionary_from_variable_list(variable_list, 
                                                                                      variable_name_list)

            if save_data:
                # Save numpy objects as npy 

                print("Saving np data structures as .npy files ...")
                dataset_utils.save_processed_data_as_npy(self.data_processed_folder, 
                                                         preprocess_data_dict,
                                                         split_percentage)
                assert 1 == 0 # Uncomment this if you want to stop after preprocessing and save
        else:
            print("Loading .npy files as np data structures ...")

            preprocess_data_dict = dataset_utils.load_processed_files_from_npy(self.data_processed_folder)
        
            seq_list, seq_list_rel, loss_mask_list, non_linear_obj, num_objs_in_seq, \
            seq_id_list, object_class_id_list, object_id_list, ego_vehicle_origin, num_seq_list, \
            straight_trajectories_list, curved_trajectories_list, city_ids, norm  = \
                operator.itemgetter(*variable_name_list)(preprocess_data_dict)

            if self.extra_data_train != -1:
                extra_preprocess_data_dict = dataset_utils.load_processed_files_from_npy(self.extra_data_processed_folder)
        
                ex_seq_list, ex_seq_list_rel, ex_loss_mask_list, ex_non_linear_obj, ex_num_objs_in_seq, \
                ex_seq_id_list, ex_object_class_id_list, ex_object_id_list, ex_ego_vehicle_origin, ex_num_seq_list, \
                ex_straight_trajectories_list, ex_curved_trajectories_list, ex_city_ids, ex_norm  = \
                    operator.itemgetter(*variable_name_list)(extra_preprocess_data_dict)

                num_val_files = len(ex_num_seq_list)
                ex_cum_start_idx = [0] + np.cumsum(ex_num_objs_in_seq).tolist()
                ex_seq_start_end = [(start, end) for start, end in zip(ex_cum_start_idx, ex_cum_start_idx[1:])]

                if self.split == "train":
                    
                    # Take a percentage of this validation files

                    num_files = math.floor(self.extra_data_train * num_val_files)

                    start_objs = 0
                    start_files = 0
                    end_objs = ex_seq_start_end[num_files][1]
                    end_files = num_files

                elif self.split == "val":

                    # Take the remaining validation files to validate

                    num_files = math.ceil((1-self.extra_data_train) * num_val_files)

                    start_objs = ex_seq_start_end[num_files][1]
                    start_files = num_files
                    end_objs = ex_seq_list.shape[0]
                    end_files = num_val_files
                
                ex_seq_list = ex_seq_list[start_objs:end_objs,:,:]
                ex_seq_list_rel = ex_seq_list_rel[start_objs:end_objs,:,:]
                ex_loss_mask_list = ex_loss_mask_list[start_objs:end_objs,:]
                ex_non_linear_obj = ex_non_linear_obj[start_objs:end_objs]
                ex_num_objs_in_seq = ex_num_objs_in_seq[start_files:end_files]
                ex_seq_id_list = ex_seq_id_list[start_objs:end_objs,:,:]
                ex_object_class_id_list = ex_object_class_id_list[start_objs:end_objs]
                ex_object_id_list = ex_object_id_list[start_objs:end_objs]
                ex_ego_vehicle_origin = ex_ego_vehicle_origin[start_files:end_files,:,:]
                ex_num_seq_list = ex_num_seq_list[start_files:end_files]
                ex_straight_trajectories_list = ex_straight_trajectories_list[start_files:end_files]
                ex_curved_trajectories_list = ex_curved_trajectories_list[start_files:end_files]
                ex_city_ids = ex_city_ids[start_files:end_files]
                
                # ex_norm not used at this moment

                if self.split == "train":

                    # Concatenate val info

                    seq_list = np.concatenate([seq_list,ex_seq_list],axis=0)
                    seq_list_rel = np.concatenate([seq_list_rel,ex_seq_list_rel],axis=0)
                    loss_mask_list = np.concatenate([loss_mask_list,ex_loss_mask_list],axis=0)
                    non_linear_obj = np.concatenate([non_linear_obj,ex_non_linear_obj],axis=0)
                    num_objs_in_seq = np.concatenate([num_objs_in_seq,ex_num_objs_in_seq],axis=0)
                    seq_id_list = np.concatenate([seq_id_list,ex_seq_id_list],axis=0)
                    object_class_id_list = np.concatenate([object_class_id_list,ex_object_class_id_list],axis=0)
                    ego_vehicle_origin = np.concatenate([ego_vehicle_origin,ex_ego_vehicle_origin],axis=0)
                    num_seq_list = np.concatenate([num_seq_list,ex_num_seq_list],axis=0)
                    straight_trajectories_list = np.concatenate([straight_trajectories_list,ex_straight_trajectories_list],axis=0)
                    curved_trajectories_list = np.concatenate([curved_trajectories_list,ex_curved_trajectories_list],axis=0)
                    city_ids = np.concatenate([city_ids,ex_city_ids],axis=0)

                elif self.split == "val":
                    
                    # Overwrite info
                    
                    seq_list = ex_seq_list
                    seq_list_rel = ex_seq_list_rel
                    loss_mask_list = ex_loss_mask_list
                    non_linear_obj = ex_non_linear_obj
                    num_objs_in_seq = ex_num_objs_in_seq
                    seq_id_list = ex_seq_id_list
                    object_class_id_list = ex_object_class_id_list
                    ego_vehicle_origin = ex_ego_vehicle_origin
                    num_seq_list = ex_num_seq_list
                    straight_trajectories_list = ex_straight_trajectories_list
                    curved_trajectories_list = ex_curved_trajectories_list
                    city_ids = ex_city_ids

        ## Create torch data

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_gt = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_gt_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_obj = torch.from_numpy(non_linear_obj).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_objs_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_id_list = torch.from_numpy(seq_id_list).type(torch.float)
        self.object_class_id_list = torch.from_numpy(object_class_id_list).type(torch.float)
        self.object_id_list = torch.from_numpy(object_id_list).type(torch.float)
        self.ego_vehicle_origin = torch.from_numpy(ego_vehicle_origin).type(torch.float)
        self.city_ids = torch.from_numpy(city_ids).type(torch.float)

        self.num_seq_list = torch.from_numpy(num_seq_list).type(torch.int)
        self.num_seq = len(num_seq_list)

        self.straight_trajectories_list = torch.from_numpy(straight_trajectories_list).type(torch.int)
        self.curved_trajectories_list = torch.from_numpy(curved_trajectories_list).type(torch.int)
        self.curved_aux_list = copy.deepcopy(self.curved_trajectories_list)
        self.norm = torch.from_numpy(np.array(norm))
        
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        """
        N.B. index represents the index of the dataloader (from 0 to N elements of the corresponding
        split). On the other hand, trajectory_index represents the number of the trajectory (csv)
        regarding the Argoverse 1.1. dataset (e.g. the scene 35.csv may be identified with the index
        32 because maybe there are not 34 csvs before this one)
        """

        global data_imgs_folder, APPLY_DATA_AUGMENTATION, PHYSICAL_CONTEXT
        data_imgs_folder = os.path.join(self.root_folder,self.split,"data_images")

        APPLY_DATA_AUGMENTATION = self.data_augmentation
        PHYSICAL_CONTEXT = self.physical_context

        if self.class_balance >= 0.0: # Only during training
            if self.cont_seqs % self.batch_size == 0: # Get a new batch
                self.cont_straight_traj = []
                self.cont_curved_traj = []

            if self.cont_seqs % self.batch_size == (self.batch_size-1):
                assert len(self.cont_straight_traj) <= self.class_balance*self.batch_size

            trajectory_index = self.num_seq_list[index]
            straight_traj = True

            if trajectory_index in self.curved_trajectories_list:
                straight_traj = False
                
            if straight_traj:
                if len(self.cont_straight_traj) >= int(self.class_balance*self.batch_size):
                    # Take a random curved trajectory from the dataset

                    aux_index = random.choice(self.curved_aux_list)
                    
                    # Update index to be included in the batch

                    index = int(np.where(self.num_seq_list == aux_index)[0])
                    self.cont_curved_traj.append(index)

                    # Remove that index from auxiliar list

                    self.curved_aux_list = \
                        self.curved_aux_list[torch.where(self.curved_aux_list != aux_index)]
                    
                    ## We run out all curves. Delete variable and re-initialize the list. Doing this
                    ## we ensure that the training runs over all possible curved (non-linear) trajectories

                    if len(self.curved_aux_list) == 0: 
                        del self.curved_aux_list
                        self.curved_aux_list = copy.deepcopy(self.curved_trajectories_list)
                else:
                    self.cont_straight_traj.append(index)
            else:
                self.cont_curved_traj.append(index)

        start, end = self.seq_start_end[index]

        try: # Bad done (for train and val) -> origin is N x 1 x 2
            out = [
                    self.obs_traj[start:end, :, :], self.pred_traj_gt[start:end, :, :],
                    self.obs_traj_rel[start:end, :, :], self.pred_traj_gt_rel[start:end, :, :],
                    self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                    self.seq_id_list[start:end, :, :], self.object_class_id_list[start:end], 
                    self.object_id_list[start:end], self.city_ids[index], self.ego_vehicle_origin[index,:,:],
                    self.num_seq_list[index], self.norm
                ] 
        except: # Well done (for test) -> origin is N x 2
            out = [
                    self.obs_traj[start:end, :, :], self.pred_traj_gt[start:end, :, :],
                    self.obs_traj_rel[start:end, :, :], self.pred_traj_gt_rel[start:end, :, :],
                    self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                    self.seq_id_list[start:end, :, :], self.object_class_id_list[start:end], 
                    self.object_id_list[start:end], self.city_ids[index], self.ego_vehicle_origin[index,:], # HERE THE DIFFERENCE
                    self.num_seq_list[index], self.norm
                ]

        # Increase file count
        
        self.cont_seqs += 1

        return out


