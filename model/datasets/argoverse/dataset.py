#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Argoverse dataset

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import random
import os
import time
import operator
import pdb

# DL & Math imports

import numpy as np
import torch
from torch.utils.data import Dataset

# Custom imports

import model.datasets.argoverse.dataset_utils as dataset_utils
import model.datasets.argoverse.geometric_functions as geometric_functions
import model.datasets.argoverse.data_augmentation_functions as data_augmentation_functions

#######################################

# Data augmentation variables

APPLY_DATA_AUGMENTATION = False

decision = [0,1] # Not apply/apply
dropout_prob = [0.1,0.9] # Not applied/applied probability
gaussian_noise_prob = [0.2,0.8]
rotation_prob = [0.5,0.5]

rotation_angles = [90,180,270]
rotation_angles_prob = [0.33,0.33,0.34]

# Auxiliar variables

data_imgs_folder = None
PHYSICAL_CONTEXT = "Dummy"

frames_path = None
dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

#######################################

def seq_collate(data):
    """
    """

    start = time.time()

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_obj, loss_mask, seq_id_list, object_class_id_list, 
     object_id_list, city_id, ego_vehicle_origin, num_seq_list, norm) = zip(*data)

    batch_size = len(ego_vehicle_origin) # tuple of tensors

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
    id_frame = torch.cat(seq_id_list, dim=0).permute(2, 0, 1) # seq_len - objs_in_curr_seq - 3

    ## Data augmentation

    curr_split = data_imgs_folder.split('/')[-3]

    if APPLY_DATA_AUGMENTATION and curr_split == "train":
        num_obstacles = obs_traj.shape[1]
        obs_len = obs_traj.shape[0]

        apply_dropout = np.random.choice(decision,num_obstacles,p=dropout_prob)
        apply_gaussian_noise = np.random.choice(decision,num_obstacles,p=gaussian_noise_prob)
        apply_rotation = np.random.choice(decision,1,p=rotation_prob) # To the whole sequence
        if apply_rotation:
            angle = np.random.choice(rotation_angles,1,p=rotation_angles_prob)

        obs_traj = data_augmentation_functions.erase_points_collate(obs_traj,apply_dropout,num_obs=obs_len,percentage=0.3)
        obs_traj = data_augmentation_functions.add_gaussian_noise_collate(obs_traj,apply_gaussian_noise,num_obstacles,num_obs=obs_len,mu=0,sigma=0.5)

        # Get new relatives

        obs_traj_rel = torch.zeros((obs_traj.shape))

        for i in range(num_obstacles): # TODO: Do this with a matricial operation
            _obs_traj = obs_traj[:,i,:]
            _obs_traj_rel = torch.zeros((_obs_traj.shape))
            _obs_traj_rel[1:,:] = _obs_traj[1:,:] - _obs_traj[:-1,:]

            obs_traj_rel[:,i,:] = _obs_traj_rel

    ## Get physical information (image or goal points. Otherwise, use dummies)

    start = time.time()

    first_obs = obs_traj[0,:,:] # 1 x agents · batch_size x 2

    # TODO: Merge load_goal_points and load_images in a single function
    
    if PHYSICAL_CONTEXT == "visual": # batch_size x channels x height x width
        frames = dataset_utils.load_images(num_seq_list, obs_traj_rel, first_obs, city_id, ego_vehicle_origin,
                                           dist_rasterized_map, object_class_id_list, data_imgs_folder,
                                           debug_images=False)
        frames = torch.from_numpy(frames).type(torch.float32)
        frames = frames.permute(0, 3, 1, 2)
    elif PHYSICAL_CONTEXT == "goals": # batch_size x num_goal_points x 2 (x|y) (real-world coordinates (HDmap))
        frames = dataset_utils.load_goal_points(num_seq_list, obs_traj_rel, first_obs, city_id, ego_vehicle_origin,
                                                dist_rasterized_map, object_class_id_list, data_imgs_folder,
                                                debug_images=False)
        frames = torch.from_numpy(frames).type(torch.float32)
    else: # dummy frames
        frames = np.random.randn(1,1,1,1)
        frames = torch.from_numpy(frames).type(torch.float32)

    end = time.time()
    # print(f"Time consumed by load_images function: {end-start}\n")

    object_cls = torch.cat(object_class_id_list, dim=0)
    obj_id = torch.cat(object_id_list, dim=0)
    ego_vehicle_origin = torch.stack(ego_vehicle_origin)
    num_seq_list = torch.stack(num_seq_list)
    norm = torch.stack(norm)

    out = [obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
           loss_mask, seq_start_end, frames, object_cls, obj_id, ego_vehicle_origin, num_seq_list, norm]

    end = time.time()
    # print(f"Time consumed by seq_collate function: {end-start}\n")

    return tuple(out)

def process_window_sequence(idx, frame_data, frames, seq_len, pred_len, 
                            file_id, split, obs_origin, 
                            rot_angle=None, augs=None):
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
        seq_len (int)
        pred_len (int)
        threshold (float)
        file_id (int)
        split (str: "train", "val", "test") 
    Output:
        num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, curr_seq_rel, 
        id_frame_list, object_class_list, city_id, ego_origin
    """

    # Prepare current sequence and get unique obstacles

    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # Unique IDs in the sequence
    agent_indeces = np.where(curr_seq_data[:, 1] == 1)[0]
    obs_len = seq_len - pred_len

    # Initialize variables

    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len)) # peds_in_curr_seq x 2 (x,y) x seq_len (ej: 50)                              
    curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len)) # peds_in_curr_seq x 2 (x,y) x seq_len (ej: 50)
    curr_loss_mask = np.zeros((len(peds_in_curr_seq), seq_len)) # peds_in_curr_seq x seq_len (ej: 50)
    object_class_list = np.zeros(len(peds_in_curr_seq)) 
    id_frame_list  = np.zeros((len(peds_in_curr_seq), 3, seq_len))

    num_objs_considered = 0
    _non_linear_obj = []
    ego_origin = [] # NB: This origin may not be the "ego" origin (that is, the AV origin). At this moment it is the
                    # obs_len-1 th absolute position of the AGENT (object of interest)
    city_id = curr_seq_data[0,5]

    # Get origin of this sequence. We assume we are going to take the AGENT as reference (object of interest in 
    # Argoverse 1.0). In the code it is written ego_vehicle but actually it is NOT the ego-vehicle (AV, which 
    # captures the scene), but another object of interest to be predicted. TODO: Change ego_vehicle_origin 
    # notation to just origin

    aux_seq = curr_seq_data[curr_seq_data[:, 2] == 1, :] # 1 is the object class, the AGENT id may not be 1!
    ego_vehicle = aux_seq[obs_origin-1, 3:5] # x,y
    ego_origin.append(ego_vehicle)

    # Iterate over all unique objects

    ## Sequence rotation

    rotate_seq = 0

    # print(">>>>>>>>>>>>>>>>>>>>>> file id: ", file_id)

    # if split == "train":
    #     if not rot_angle:
    #         rotate_seq = 0
    #     elif rot_angle in rotation_available_angles:
    #         rotate_seq = 1
    #         rotation_angle = rot_angle
    #     elif rot_angle == -1:
    #         rotate_seq = np.random.randint(2,size=1) # Rotate the sequence (so, the image) if 1
    #         if rotate_seq:
    #             availables_angles = [90,180,270]
    #             rotation_angle = availables_angles[np.random.randint(3,size=1).item()]

    #         rotate_seq = 1
    #         rotation_angle = rot_angle
    
    # print("num objs: ", len(peds_in_curr_seq))

    for index, ped_id in enumerate(peds_in_curr_seq):
        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

        # curr_ped_seq = np.around(curr_ped_seq, decimals=4)
        #################################################################################
        ## test
        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
        #################################################################################
        if (pad_end - pad_front != seq_len) or (curr_ped_seq.shape[0] != seq_len): # If the object has less than "seq_len" observations,
                                                                                   # it is discarded
            continue
        # Determine if data aug will be applied or not

        # if split == "train":
        #     data_aug_flag = np.random.randint(2)
        # else:
        #     data_aug_flag = 0

        # Get object class id

        object_class_list[num_objs_considered] = curr_ped_seq[0,2] # 0 == AV, 1 == AGENT, 2 == OTHER

        # Record seqname, frame and ID information

        cache_tmp = np.transpose(curr_ped_seq[:,:2])
        id_frame_list[num_objs_considered, :2, :] = cache_tmp
        id_frame_list[num_objs_considered,  2, :] = file_id

        # Get x-y data (w.r.t the sequence origin, so they are absolute 
        # coordinates but in the local frame, not map (global) frame)
        
        curr_ped_seq = np.transpose(curr_ped_seq[:, 3:5])
        curr_ped_seq = curr_ped_seq - ego_origin[0].reshape(-1,1)
        first_obs = curr_ped_seq[:,0].reshape(-1,1)

        # Rotation (If the image is rotated, all trajectories must be rotated)

        # rotate_seq = 0

        # if rotate_seq:
        #     curr_ped_seq = dataset_utils.rotate_traj(curr_ped_seq,rotation_angle)

        # data_aug_flag = 0

        # if split == "train" and (data_aug_flag == 1 or augs):
        #     # Add data augmentation

        #     if not augs:
        #         print("Get comb")
        #         augs = dataset_utils.get_data_aug_combinations(3) # Available data augs: Swapping, Erasing, Gaussian noise

        #     ## 1. Swapping

        #     if augs[0]:
        #         print("Swapping")
        #         curr_ped_seq = dataset_utils.swap_points(curr_ped_seq,num_obs=obs_len)

        #     ## 2. Erasing

        #     if augs[1]:
        #         print("Erasing")
        #         curr_ped_seq = dataset_utils.erase_points(curr_ped_seq,num_obs=obs_len,percentage=0.3)

        #     ## 3. Add Gaussian noise

        #     if augs[2]:
        #         print("Gaussian")
        #         curr_ped_seq = dataset_utils.add_gaussian_noise(curr_ped_seq,num_obs=obs_len,mu=0,sigma=0.5)

        # Make coordinates relative (relative here means displacements between consecutive steps)

        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape) 
        rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # Get displacements between consecutive steps
        
        _idx = num_objs_considered
        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

        # Linear vs Non-Linear Trajectory
        if split != 'test':
            # non_linear = _non_linear_obj.append(poly_fit(curr_ped_seq, pred_len, threshold))
            try:
                non_linear = geometric_functions.get_non_linear(file_id, curr_seq, idx=_idx, obj_kind=curr_ped_seq[0,2],
                                                                threshold=2, debug_trajectory_classifier=False)
            except: # E.g. All max_trials iterations were skipped because each randomly chosen sub-sample 
                    # failed the passing criteria. Return non-linear because RANSAC could not fit a model
                non_linear = 1.0
            _non_linear_obj.append(non_linear)
        curr_loss_mask[_idx, pad_front:pad_end] = 1

        # Add num_objs_considered
        num_objs_considered += 1

    return num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, curr_seq_rel, \
           id_frame_list, object_class_list, city_id, ego_origin
           
class ArgoverseMotionForecastingDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, dataset_name, root_folder, obs_len=20, pred_len=30, distance_threshold=30,
                 split='train', split_percentage=0.1, start_from_percentage=0.0, shuffle=False, 
                 batch_size=16, class_balance=-1.0, obs_origin=1, preprocess_data=False, save_data=False, 
                 data_augmentation=False, physical_context="dummy"):
        super(ArgoverseMotionForecastingDataset, self).__init__()

        # Initialize self variables

        self.dataset_name = dataset_name
        self.root_folder = root_folder
        data_processed_folder = root_folder + split + "/data_processed_" + str(int(split_percentage*100)) + "_percent"

        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.distance_threshold = distance_threshold # Monitorize distance_threshold around the AGENT
        self.split = split
        self.batch_size = batch_size
        self.class_balance = class_balance
        self.obs_origin = obs_origin
        self.min_objs = 2 # Minimum number of objects to include the scene (AV and AGENT)
        self.cont_seqs = 0

        variable_name_list = ['seq_list','seq_list_rel','loss_mask_list','non_linear_obj',
                              'num_objs_in_seq','seq_id_list','object_class_id_list',
                              'object_id_list','ego_vehicle_origin','num_seq_list',
                              'straight_trajectories_list','curved_trajectories_list','city_id','norm'] 

        PREPROCESS_DATA = preprocess_data
        SAVE_DATA = save_data
        global APPLY_DATA_AUGMENTATION, PHYSICAL_CONTEXT
        APPLY_DATA_AUGMENTATION = data_augmentation
        PHYSICAL_CONTEXT = physical_context

        # Preprocess data (from raw .csvs to torch Tensors, at least including the 
        # AGENT (most important vehicle) and AV

        if PREPROCESS_DATA and not os.path.isdir(data_processed_folder):
            folder = root_folder + split + "/data/"
            files, num_files = dataset_utils.load_list_from_folder(folder)

            # Sort list and apply shuffling/analyze a specific percentage/from a specific position

            file_id_list, root_file_name = dataset_utils.get_sorted_file_id_list(files)     
            file_id_list = dataset_utils.apply_shuffling_percentage_startfrom(file_id_list, num_files,
                                                                              shuffle=shuffle, 
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

            t0 = time.time()
            for i, file_id in enumerate(file_id_list):
                t1 = time.time()
                print(f"File {file_id} -> {i}/{len(file_id_list)}")
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
                                            self.seq_len, self.pred_len, 
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
                # print("File {} consumed {} s".format(i, (time.time() - t1))) 

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

            ## normalize abs and relative data # TODO: Is this used now?
            abs_norm = (seq_list.min(), seq_list.max())
            # seq_list = (seq_list - seq_list.min()) / (seq_list.max() - seq_list.min())

            rel_norm = (seq_list_rel.min(), seq_list_rel.max())
            # seq_list_rel = (seq_list_rel - seq_list_rel.min()) / (seq_list_rel.max() - seq_list_rel.min())
            norm = (abs_norm, rel_norm)

            # Create dictionary with all processed data

            variable_list = [seq_list, seq_list_rel, loss_mask_list, non_linear_obj, num_objs_in_seq,
                             seq_id_list, object_class_id_list, object_id_list, ego_vehicle_origin, 
                             num_seq_list, straight_trajectories_list, curved_trajectories_list, city_ids, norm]
            preprocess_data_dict = dataset_utils.create_dictionary_from_variable_list(variable_list, 
                                                                                      variable_name_list)

            if SAVE_DATA:
                # Save numpy objects as npy 

                print("Saving np data structures as .npy files ...")
                dataset_utils.save_processed_data_as_npy(data_processed_folder, 
                                                         preprocess_data_dict,
                                                         split_percentage)
                #assert 1 == 0 # Uncomment this if you want to preprocess -> save -> 
                              # -> continue with the dataset (for example in training)
        else:
            print("Loading .npy files as np data structures ...")

            preprocess_data_dict = dataset_utils.load_processed_files_from_npy(data_processed_folder)
        
            seq_list, seq_list_rel, loss_mask_list, non_linear_obj, num_objs_in_seq, \
            seq_id_list, object_class_id_list, object_id_list, ego_vehicle_origin, num_seq_list, \
            straight_trajectories_list, curved_trajectories_list, city_ids, norm  = \
                operator.itemgetter(*variable_name_list)(preprocess_data_dict)

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
        self.norm = torch.from_numpy(np.array(norm))
        
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        global data_imgs_folder
        data_imgs_folder = self.root_folder + self.split + "/data_images/"

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

                    aux_index = random.choice(self.curved_trajectories_list)
                    index = int(np.where(self.num_seq_list == aux_index)[0])
                    self.cont_curved_traj.append(index)
                else:
                    self.cont_straight_traj.append(index)
            else:
                self.cont_curved_traj.append(index)

        start, end = self.seq_start_end[index]

        out = [
                self.obs_traj[start:end, :, :], self.pred_traj_gt[start:end, :, :],
                self.obs_traj_rel[start:end, :, :], self.pred_traj_gt_rel[start:end, :, :],
                self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                self.seq_id_list[start:end, :, :], self.object_class_id_list[start:end], 
                self.object_id_list[start:end], self.city_ids[index], self.ego_vehicle_origin[index,:,:],
                self.num_seq_list[index], self.norm
              ] 

        # Increase file count
        
        self.cont_seqs += 1

        return out


