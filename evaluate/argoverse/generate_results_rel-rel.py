#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Generate results:
## If 'train' or 'val': Check the results with the best ckpt (in order to obtain qualitative results, check metrics, etc.)
## If 'test': Generate the .h5 file to submit to the Argoverse 1.1 Motion Forecasting Leaderboard (qualitative results are orientative
##            since we do not have the groundtruth for the target agent)

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import argparse
import os
import sys
import yaml
import pdb
import time
import glob
import csv
import pandas as pd
import git

from datetime import datetime
from pathlib import Path
from prodict import Prodict

# DL & Math imports

import math
import numpy as np
import torch

from torch.utils.data import DataLoader

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.plot_functions as plot_functions
import model.datasets.argoverse.dataset_utils as dataset_utils
import model.datasets.argoverse.data_augmentation_functions as data_augmentation_functions

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
from model.utils.checkpoint_data import get_generator
from model.trainers.trainer_mapfe4mp import cal_ade_multimodal, cal_fde_multimodal

from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.map_representation.map_api import ArgoverseMap

#######################################

config = None

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument("--device_gpu", required=True, default=0, type=int)
parser.add_argument("--split", required=True, default="val", type=str)
parser.add_argument("--batch_size", required=True, default=1, type=int)

avm = ArgoverseMap()

LIMIT_FILES = -1 # From 1 to num_files.
                 # -1 by default to analyze all files of the specified split percentage

OBS_ORIGIN = 20
PRED_LEN = 30
ARGOVERSE_NUM_MODES = 6
DATA_DIM = 2

GENERATE_QUALITATIVE_RESULTS = True
PLOT_WORST_SCENES = False
LIMIT_QUALITATIVE_RESULTS = 300
DEBUG = False
COMPUTE_METRICS = True
PLOT_METRICS = False

def generate_csv(results_path, ade_um_list, fde_um_list, ade_mm_list, fde_mm_list, 
                 num_seq_list, traj_kind_list, sort=False):
    """
    If sort = True, sort by ADE
    """

    now = datetime.now()

    if sort:
        results_path = os.path.join(results_path,"metrics_sorted_ade.csv")  
    else:
        results_path = os.path.join(results_path,"metrics.csv")  

    mean_ade_um = round(sum(ade_um_list) / (len(ade_um_list)),3)
    mean_fde_um = round(sum(fde_um_list) / (len(fde_um_list)),3)
    
    mean_ade_mm = round(sum(ade_mm_list) / (len(ade_mm_list)),3)
    mean_fde_mm = round(sum(fde_mm_list) / (len(fde_mm_list)),3)

    with open(results_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        header = ['index','seq_csv','curve_straight','ade_k_1','fde_k_1','ade_k_6','fde_k_6']
        csv_writer.writerow(header)

        if sort:
            indeces = np.argsort(ade_mm_list)
        else:
            indeces = np.arange(len(ade_mm_list))

        for _,index in enumerate(indeces):
            traj_kind = traj_kind_list[index]
            seq_id = num_seq_list[index] 
            curr_ade_um = round(ade_um_list[index],3)
            curr_fde_um = round(fde_um_list[index],3)
            curr_ade_mm = round(ade_mm_list[index],3)
            curr_fde_mm = round(fde_mm_list[index],3)

            data = [str(index),str(seq_id),traj_kind,curr_ade_um,curr_fde_um,curr_ade_mm,curr_fde_mm]

            csv_writer.writerow(data)

        # Write mean ADE and FDE 

        csv_writer.writerow(['-','-','-','-','-','-','-'])
        csv_writer.writerow(['-','-','-',mean_ade_um,mean_fde_um,mean_ade_mm,mean_fde_mm])

def get_worst_scenes(csv_file_sorted):
    """
    """

    df = pd.read_csv(csv_file_sorted,sep=" ")
    seq_csvs_column = df["Index"][-LIMIT_QUALITATIVE_RESULTS-2:-2] # Index from 1 to num_files, not seq.csv!
    worst_scenes = np.array(seq_csvs_column).tolist()
    worst_scenes = list(map(int,worst_scenes))

    return worst_scenes

def evaluate(loader, generator, config, split, current_cuda, pred_len, results_path, worst_scenes=None):
    """
    """
    # assert loader.batch_size == 1

    output_predictions = {}
    output_probabilities = {}
    
    ade_mm_list = [] # Multimodal, store min ADE out of k modes
    fde_mm_list = [] # Multimodal, store min FDE out of k modes
    ade_um_list = [] # Unimodal, store ADE of the mode with the highest confidence
    fde_um_list = [] # Unimodal, store FDE of the mode with the highest confidence
    
    traj_kind_list = []
    num_seq_list = []
    plot_scene = 0

    data_folder = BASE_DIR+f"/data/datasets/argoverse/motion-forecasting/{split}/data/"
    file_list = glob.glob(os.path.join(data_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    num_files = len(file_list)

    print("Data folder: ", data_folder)
    print("Num files ", num_files)

    time_per_iteration = float(0)
    aux_time = float(0)

    batch_size = config.dataset.batch_size
    
    with torch.no_grad(): # During inference (regardless the split), gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            if LIMIT_FILES != -1: 
                if (batch_index+1)*batch_size > LIMIT_FILES:
                    break

                files_remaining = LIMIT_FILES - (batch_index+1)*batch_size
                print(f"Evaluating file {(batch_index+1)*batch_size}/{LIMIT_FILES}")
            else: 
                files_remaining = num_files - (batch_index+1)*batch_size
                print(f"Evaluating file {(batch_index+1)*batch_size}/{len(loader)*batch_size}")
            batch_remaining = files_remaining / batch_size
            
            if worst_scenes: # Analyze some specific sequences
                if batch_index > max(worst_scenes) or plot_scene >= LIMIT_QUALITATIVE_RESULTS:
                    break
                elif batch_index not in worst_scenes:
                    continue
            
            start = time.time()
        
            # Load batch in device

            if config.hyperparameters.physical_context != "plausible_centerlines+area":
                # Here the physical info is a single tensor

                batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
                 loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm, 
                 target_agent_orientation, relevant_centerlines) = batch
                
            elif config.hyperparameters.physical_context == "plausible_centerlines+area":

                # TODO: In order to improve this, seq_collate should return a dictionary instead of a tuple
                # in order to avoid hardcoded positions
                phy_info = batch[-1] # phy_info should be in the last position!

                batch = [tensor.cuda(current_cuda) for tensor in batch if torch.is_tensor(tensor)]

                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
                 loss_mask, seq_start_end, object_cls, obj_id, map_origin, num_seq, norm, 
                 target_agent_orientation) = batch

            ## Get AGENT (most interesting obstacle) id

            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # Get predictions

            # TODO: Not used at this moment
            if config.hyperparameters.num_modes == 1: # The model is unimodal # TODO: -> conf here is 1/6

                pred_traj_fake_rel_list = []
                pred_traj_fake_list = []

                for _ in range(ARGOVERSE_NUM_MODES):
                    ## Get predictions (relative displacements)
                    
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx)
                    pred_traj_fake_rel_list.append(pred_traj_fake_rel)

                    ## Get predictions in absolute -> map coordinates

                    pred_traj_fake = dataset_utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1, agent_idx, :])
                    pred_traj_fake_list.append(pred_traj_fake)

                pred_traj_fake_rel = torch.stack(pred_traj_fake_rel_list, axis=0).view(-1, ARGOVERSE_NUM_MODES, DATA_DIM) # pred_len x num_modes x data_dim
                pred_traj_fake_global = torch.stack(pred_traj_fake_list + map_origin, axis=0).view(ARGOVERSE_NUM_MODES,-1,DATA_DIM) # num_modes x pred_len x data_dim
            
                conf_array = np.ones([ARGOVERSE_NUM_MODES])/ARGOVERSE_NUM_MODES

            else: # The model is multimodal
                assert config.hyperparameters.num_modes == ARGOVERSE_NUM_MODES

                if config.hyperparameters.physical_context == "plausible_centerlines+area":
                    # TODO: Encapsulate this as a function
       
                    # Plausible area (discretized)

                    plausible_area_array = np.array([curr_phy_info["plausible_area_abs"] for curr_phy_info in phy_info]).astype(np.float32)
                    plausible_area = torch.from_numpy(plausible_area_array).type(torch.float).cuda(obs_traj.device)

                    # Compute relevant centerlines

                    num_relevant_centerlines = [len(curr_phy_info["relevant_centerlines_abs"]) for curr_phy_info in phy_info]
                    aux_indeces_list = []
                    for num_ in num_relevant_centerlines:
                        if num_ >= config.hyperparameters.num_modes: # Limit to the first config.hyperparameters.num_modes centerlines. We assume
                                                            # the oracle is here. TODO: Force this from the preprocessing 
                            aux_indeces = np.arange(config.hyperparameters.num_modes)
                            aux_indeces_list.append(aux_indeces)
                        else: # We have less centerlines than config.hyperparameters.num_modes
                            quotient = int(config.hyperparameters.num_modes / num_)
                            remainder = config.hyperparameters.num_modes % num_

                            aux_indeces = np.arange(num_)
                            aux_indeces = np.tile(aux_indeces,quotient)

                            if remainder != 0:
                                aux_indeces_ = np.arange(remainder)
                                aux_indeces = np.hstack((aux_indeces,aux_indeces_))
                            
                            assert len(aux_indeces) == config.hyperparameters.num_modes, "Number of centerlines does not match the number of required modes"
                            aux_indeces_list.append(aux_indeces)

                    centerlines_indeces = np.array(aux_indeces_list)

                    relevant_centerlines = []
                    for mode in range(config.hyperparameters.num_modes):

                        # Dynamic map info (one centerline per iteration -> Max Num_Modes iterations)

                        current_centerlines_indeces = centerlines_indeces[:,mode]
                        current_centerlines_list = []
                        
                        for i in range(batch_size):
                            current_centerline = phy_info[i]["relevant_centerlines_abs"][current_centerlines_indeces[i]]
                            current_centerlines_list.append(current_centerline)

                        current_centerlines = torch.from_numpy(np.array(current_centerlines_list)).type(torch.float).cuda(obs_traj.device)
                        relevant_centerlines.append(current_centerlines.unsqueeze(0))

                    relevant_centerlines = torch.cat(relevant_centerlines,dim=0)

                    ## Get predictions (relative displacements)

                    pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=plausible_area, relevant_centerlines=relevant_centerlines)
                else:
                    pred_traj_fake_rel, conf = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx, relevant_centerlines=relevant_centerlines)
                    
                ## Get predictions: relative -> absolute -> map coordinates

                pred_traj_fake = dataset_utils.relative_to_abs_multimodal(pred_traj_fake_rel, obs_traj[-1,agent_idx,:])
                pred_traj_fake_global = pred_traj_fake + map_origin.unsqueeze(1).unsqueeze(1)
                
            # Hereafter, we have multimodality (either by iterating the same model K modes or 
            # returning K modes in the same forward)
            
            # Iterate over the whole batch to compute the metrics and plot qualitative results
            
            # Transform to CPU (to acelerate the rotation and plot qualitative results)

            obs_traj = obs_traj.cpu()
            pred_traj_gt = pred_traj_gt.cpu()
            pred_traj_fake = pred_traj_fake.cpu()
            conf = conf.cpu()
            
            target_agent_orientation = target_agent_orientation.cpu()
            map_origin = map_origin.cpu()
            relevant_centerlines = relevant_centerlines.cpu()  
            non_linear_obj = non_linear_obj.cpu()
            object_cls = object_cls.cpu()
            
            for i in range(len(num_seq)):
            # OBS: We cannot iterate always over batch_size length, since at the
            # end our batch will usually have less elements than batch_size (e.g. 560 vs 1024)
            # So, we must make sure how many elements our current batch has 
             
                # Get seq_id and city for this sequence

                seq_id = num_seq[i].cpu().item()
                if DEBUG: print(f"{seq_id}.csv")
                
                path = os.path.join(data_folder,str(seq_id)+".csv")
                data = dataset_utils.read_file(path) 
                _, city_name = dataset_utils.get_origin_and_city(data,OBS_ORIGIN)  

                start_seq, end_seq = seq_start_end[i]
                
                ade_min, fde_min = None, None
            
                if config.dataset.apply_rotation: # We have to counter-rotate in order to have again the original sequence
                    yaw_aux = (math.pi/2 - target_agent_orientation[i]) 
                    c, s = torch.cos(yaw_aux), torch.sin(yaw_aux)
                    R = torch.tensor([[c,-s],  # Rot around the map z-axis
                                      [s, c]])

                    obs_traj[:,start_seq:end_seq,:] = data_augmentation_functions.rotate_traj(obs_traj[:,start_seq:end_seq,:],R)
                    pred_traj_gt[:,start_seq:end_seq,:] = data_augmentation_functions.rotate_traj(pred_traj_gt[:,start_seq:end_seq,:],R)
                    pred_traj_fake[i] = data_augmentation_functions.rotate_traj(pred_traj_fake[i],R)
                    map_origin[i] = data_augmentation_functions.rotate_traj(map_origin[i],R)
                    
                    if config.hyperparameters.physical_context == "social":
                        relevant_centerlines = torch.tensor([])
                    relevant_centerlines[i] = data_augmentation_functions.rotate_traj(relevant_centerlines[i],R)

                    pred_traj_fake_global[i] = pred_traj_fake[i] + map_origin[i]
                
                if COMPUTE_METRICS and split != "test":
                    agent_pred_gt = pred_traj_gt[:,agent_idx[i],:] # pred_len (30) x 1 (agent) x data_dim (2) -> "Abs" coordinates (around 0,0)
                    agent_pred_gt = agent_pred_gt.unsqueeze(0).permute(1,0,2) # pred_len x batch_size x data_dim
                    agent_pred_fake = pred_traj_fake[i]
                    agent_pred_fake = agent_pred_fake.unsqueeze(0) # batch_size x num_modes x pred_len x data_dim

                    agent_non_linear_obj = non_linear_obj[agent_idx[i]]
                    agent_linear_obj = 1 - agent_non_linear_obj

                    agent_obj_id = obj_id[agent_idx[i]]
                    agent_mask = np.where(agent_obj_id.cpu() == -1, 0, 1)
                    
                    ade, ade_min = cal_ade_multimodal(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)
                    fde, fde_min = cal_fde_multimodal(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)
                    
                    # Assuming the mode with the best confidence (k = 1)
                    
                    curr_conf = conf[i]
                    mode_max_conf = torch.argmax(curr_conf)
                    ade_unimodal = ade[0,mode_max_conf] / pred_len
                    fde_unimodal = fde[0,mode_max_conf]
                    
                    if DEBUG: print(f"k = 1 -> ADE, FDE: ", ade_unimodal, fde_unimodal)
                    
                    # Considering k modes
                    
                    ade_min = ade_min.item() / pred_len
                    fde_min = fde_min.item()
                    if DEBUG: print(f"k = {ARGOVERSE_NUM_MODES} -> ADE min, FDE min: ", ade_min, fde_min)

                    # Store for csv
                    
                    ade_um_list.append(ade_unimodal)
                    fde_um_list.append(fde_unimodal)
                    ade_mm_list.append(ade_min)
                    fde_mm_list.append(fde_min)
                    num_seq_list.append(seq_id)
                    
                    # If the agent conducts a curve (these trajectories are suppossed to be harder -> Higher ADE and FDE)

                    if agent_non_linear_obj.item(): traj_kind_list.append(1) # Curved trajectory
                    else: traj_kind_list.append(0) # The agent conducts a curve

                ## (Optional) Plot results

                if GENERATE_QUALITATIVE_RESULTS and plot_scene < LIMIT_QUALITATIVE_RESULTS:
                    plot_scene += 1

                    curr_map_origin = map_origin[i]
                    curr_object_class_id_list = object_cls[start_seq:end_seq]

                    # Argoverse standard plot

                    if config.hyperparameters.physical_context == "dummy" or config.hyperparameters.physical_context == "social":
                        relevant_centerlines_abs = []
                    else:
                        # relevant_centerlines_abs = relevant_centerlines[i].cpu().numpy()
                        relevant_centerlines_abs = relevant_centerlines[i].unsqueeze(0).cpu().numpy() # TODO: Check this

                    if not PLOT_METRICS:
                        ade_min, fde_min = None, None
                        
                    plot_functions.viz_predictions_all(seq_id,
                                                       results_path,
                                                       obs_traj[:,start_seq:end_seq,:].permute(1,0,2).numpy(), # All obstacles
                                                       pred_traj_fake[i].squeeze(0).numpy(), # Only AGENT (MM prediction)
                                                       conf[i].unsqueeze(0).numpy(), # Only AGENT (MM confidence) # TODO: Check this
                                                       target_agent_orientation[i].numpy(),
                                                       pred_traj_gt[:,start_seq:end_seq,:].permute(1,0,2).numpy(), # All obstacles                                                  
                                                       curr_object_class_id_list.numpy(),
                                                       city_name,
                                                       curr_map_origin.numpy(),
                                                       avm,
                                                       dist_rasterized_map=50,
                                                       relevant_centerlines_abs=relevant_centerlines_abs,
                                                       save=True,
                                                       ade_metric=ade_min,
                                                       fde_metric=fde_min,
                                                       plot_output_confidences=True,
                                                       worst_scenes=worst_scenes)

                pred_traj_fake_global_aux = pred_traj_fake_global[i].view(ARGOVERSE_NUM_MODES, PRED_LEN, DATA_DIM)
                output_predictions[seq_id] = pred_traj_fake_global_aux.cpu().numpy()
                output_probabilities[seq_id] = conf[i].numpy().reshape(-1)

                file_list.remove(seq_id)

            end = time.time()
            aux_time += (end-start)
            time_per_iteration = aux_time/(batch_index+1)

            print(f"Time per iteration: {time_per_iteration} s. \n \
                    Estimated time to finish ({files_remaining} files): {round((time_per_iteration*batch_remaining)/60)} min")

        # Add sequences not loaded in dataset

        try:
            print("No processed files: {}".format(len(file_list)))
        except:
            print("All files have been processed")

    return output_predictions, output_probabilities, ade_um_list, fde_um_list, ade_mm_list, fde_mm_list, traj_kind_list, num_seq_list   

def main(args):
    """
    """

    # Load config file from this specific model

    model = args.model_path.split('/')[:-1]
    config_path = os.path.join(BASE_DIR,*model,"config_file.yml")

    global config
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        print(yaml.dump(config, default_flow_style=False))
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR
        config.device_gpu = args.device_gpu
    
    config.dataset.split = args.split 

    # config.dataset.split_percentage = 1.0 # To generate the final results, must be 1
    # In general, the algorithm used to generate the test results should have been trained with the whole
    # dataset (100 %)

    # TODO: FIX THIS AND ALLOW BATCH_SIZE > 1 TO REDUCE THE TIME TO GENERATE THE RESULTS
    # config.dataset.batch_size = 1 # Better to build the h5 results file
    config.dataset.batch_size = args.batch_size
    
    # assert config.dataset.batch_size == 1, "At this moment the result generator is designed for batch size 1"

    config.dataset.num_workers = 0
    config.dataset.class_balance = -1.0 # Do not consider class balance in the split test
    config.dataset.hard_mining = -1.0
    config.dataset.shuffle = False
    config.dataset.data_augmentation = False
    if config.dataset.split == "test": # In test, 0 (we do not have the gt). Otherwise, 30
        config.hyperparameters.pred_len = 0 
    else:
        config.hyperparameters.pred_len = 30

    current_cuda = torch.device(f"cuda:{config.device_gpu}")

    # Dataloader

    print(f"Load {config.dataset.split} split...")
    
    data_split = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                   root_folder=config.dataset.path,
                                                   imgs_folder=config.dataset.imgs_folder,
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split=config.dataset.split,
                                                   split_percentage=config.dataset.split_percentage,
                                                   class_balance=config.dataset.class_balance,
                                                   hard_mining=config.dataset.hard_mining,
                                                   obs_origin=config.hyperparameters.obs_origin,
                                                   apply_rotation=config.dataset.apply_rotation,
                                                   physical_context=config.hyperparameters.physical_context,
                                                   extra_data_train=config.dataset.extra_data_train,
                                                   preprocess_data=config.dataset.preprocess_data,
                                                   save_data=config.dataset.save_data)

    split_loader = DataLoader(data_split,
                              batch_size=config.dataset.batch_size,
                              shuffle=config.dataset.shuffle,
                              num_workers=config.dataset.num_workers,
                              collate_fn=seq_collate)

    # Get generator

    print("Load generator...")
    generator = get_generator(args.model_path, config)

    # Evaluate model and get metrics

    ## Get experiment name

    aux_list = args.model_path.split('/')
    model_index = aux_list.index(config.model.name)
    exp_name = os.path.join(*args.model_path.split('/')[model_index:-1]) # model_type/split_percentage/exp_name 
                                                                         # (e.g. social_lstm_mhsa/100.0_percent/exp1)

    ## Create results folder if does not exist

    results_path = os.path.join(config.base_dir,"results",exp_name,config.dataset.split)

    if not os.path.exists(results_path):
        print("Create results path folder: ", results_path)
        os.makedirs(results_path) # os.makedirs create intermediate directories. os.mkdir only the last one

    # Get worst scenes (only if metrics from split = 100 % has been previously computed) in order to plot and analyze

    worst_scenes = None
    if int(config.dataset.split_percentage * 100) == 100:
        metrics_sorted_csv = os.path.join(results_path,"metrics_sorted_ade.csv")

        if os.path.isfile(metrics_sorted_csv) and PLOT_WORST_SCENES:
            worst_scenes = get_worst_scenes(metrics_sorted_csv)

    print(f"Evaluate model in {config.dataset.split} split")
    output_predictions, output_probabilities, ade_um_list, fde_um_list, ade_mm_list, fde_mm_list, traj_kind_list, num_seq_list = \
        evaluate(split_loader, generator, config, 
                 config.dataset.split, current_cuda, 
                 config.hyperparameters.pred_len, results_path,
                 worst_scenes=worst_scenes)

    if not os.path.exists(results_path):
        print("Create results path folder: ", results_path)
        os.makedirs(results_path) # os.makedirs create intermediate directories. os.mkdir only the last one

    if config.dataset.split == "test" and config.dataset.split_percentage == 1.0:
        # Generate h5 file for Argoverse Motion-Forecasting competition (only test split)   

        generate_forecasting_h5(output_predictions, results_path, probabilities=output_probabilities)

    elif COMPUTE_METRICS and not PLOT_WORST_SCENES: # val or train

        # Write results (ade and fde with the corresponding trajectory type) in CSV

        generate_csv(results_path, ade_um_list, fde_um_list, ade_mm_list, fde_mm_list, num_seq_list, traj_kind_list)
        generate_csv(results_path, ade_um_list, fde_um_list, ade_mm_list, fde_mm_list, num_seq_list, traj_kind_list, sort=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# Best model at this moment (in terms of validation). OBS: It has hard-mining from val, it may have a leak
"""
python evaluate/argoverse/generate_results_rel-rel.py \
--model_path "save/argoverse/mapfe4mp/100.0_percent/previous_validation/test_9/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--device_gpu 1 --split "train"
"""

# Best model at this moment (in terms of test) -> 0.99, 1.67
"""
python evaluate/argoverse/generate_results_rel-rel.py \
--model_path "save/argoverse/mapfe4mp/100.0_percent/test_8/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--device_gpu 0 --split "val"
"""

"""
python evaluate/argoverse/generate_results_rel-rel.py \
--model_path "save/argoverse/mapfe4mp/100.0_percent/test_12/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--device_gpu 0 --split "val" --batch_size 1024
"""