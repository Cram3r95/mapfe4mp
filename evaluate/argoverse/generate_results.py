#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Generate results:
## If 'train' or 'val': Check the results with the best ckpt (in order to obtain qualitative results, check metrics, etc.)
## If 'test': Generate the .h5 file to submit to the Argoverse 1.1 Motion Forecasting Leaderboard (qualitative results are orientative
##            since we do not have the groundtruth for the target agent)

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
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

from pathlib import Path
from prodict import Prodict

# DL & Math imports

import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

import model.datasets.argoverse.plot_functions as plot_functions
from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
import model.datasets.argoverse.dataset_utils as dataset_utils
from model.utils.checkpoint_data import get_generator
from model.trainers.trainer_social_lstm_mhsa import cal_ade, cal_fde

from argoverse.evaluation.competition_util import generate_forecasting_h5

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--num_modes', required=True, default=6, type=int)
parser.add_argument("--device_gpu", required=True, default=0, type=int)

dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]
GENERATE_QUALITATIVE_RESULTS = True
COMPUTE_METRICS = True

def evaluate(loader, generator, num_modes, split, current_cuda, pred_len):
    """
    """

    assert loader.batch_size == 1

    output_all = {}
    ade_list = []
    fde_list = []
    traj_kind_list = []
    num_seq_list = []

    data_folder = BASE_DIR+f"/data/datasets/argoverse/motion-forecasting/{split}/data/"
    file_list = glob.glob(os.path.join(data_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    print("Data folder: ", data_folder)
    print("Num files ", len(file_list))

    time_per_iteration = float(0)
    aux_time = float(0)

    with torch.no_grad(): # During inference (regardless the split), gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print(f"Evaluating batch {batch_index+1}/{len(loader)}")

            if batch_index > 150:
                break

            files_remaining = len(file_list) - batch_index
            start = time.time()

            # Load batch in device

            batch = [tensor.cuda(current_cuda) for tensor in batch]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, map_origin, num_seq, norm) = batch

            ## Get AGENT (most interesting obstacle) id

            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # Get predictions (relative displacements)

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx) # pred_len x batch_size x 2

            ## TODO: Avoid repeating the prediction -> Use a multimodal generator!

            # pred_traj_fake_rel = pred_traj_fake_rel.unsqueeze(dim=0) # pred_len x batch_size x 2 -> batch_size x pred_len x num_modes x 2
            # pred_traj_fake_rel = torch.repeat_interleave(pred_traj_fake_rel,num_modes,dim=2) # 1,30,1,2 -> 1,30,6,2
            pred_traj_fake_rel = torch.repeat_interleave(pred_traj_fake_rel,num_modes,dim=1) # 30,1,2 -> 30,6,2

            ## (Optional) Plot results

            seq_id = num_seq.cpu().item()

            if GENERATE_QUALITATIVE_RESULTS and seq_id < 100:
                if split == "test":
                    curr_map_origin = map_origin[0]
                    curr_traj_rel = obs_traj_rel
                else:
                    curr_map_origin = map_origin[0][0] # TODO: Generate again val and train data without [[]] in the map!
                    curr_traj_rel = torch.cat((obs_traj_rel,
                                               pred_traj_gt_rel),dim=0)
                curr_first_obs = obs_traj[0,:,:] 
                curr_object_class_id_list = object_cls

                filename = f"data/datasets/argoverse/motion-forecasting/{split}/data_images/{seq_id}.png"

                plot_functions.plot_trajectories(filename,curr_traj_rel,curr_first_obs,
                                                 curr_map_origin,curr_object_class_id_list,dist_rasterized_map,
                                                 rot_angle=-1,obs_len=obs_traj.shape[0],
                                                 smoothen=False, save=True, pred_trajectories_rel=pred_traj_fake_rel)

            if COMPUTE_METRICS: # Metrics must be in absolute coordinates (either around 0,0 or around map origin)
                agent_pred_gt = pred_traj_gt[:,agent_idx,:]
                agent_pred_fake_rel = pred_traj_fake_rel[:,0,:].unsqueeze(dim=1) # TODO: At this moment all modes are the same. Compute Multimodal
                                                                # ADE/FDE evaluation when we have real multimodality
                agent_last_obs = obs_traj[-1,agent_idx,:]
                agent_pred_fake = dataset_utils.relative_to_abs(agent_pred_fake_rel, agent_last_obs) 
                agent_non_linear_obj = non_linear_obj[agent_idx]
                agent_linear_obj = 1 - agent_non_linear_obj

                agent_obj_id = obj_id[agent_idx]
                agent_mask = np.where(agent_obj_id.cpu() == -1, 0, 1)
                agent_mask = torch.tensor(agent_mask, device=agent_obj_id.device).reshape(-1)
                
                ade, ade_l, ade_nl = cal_ade(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)

                fde, fde_l, fde_nl = cal_fde(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)

                ade = ade.item() / pred_len
                fde = fde.item()

                ade_list.append(ade)
                fde_list.append(fde)
                num_seq_list.append(seq_id)

                # If the agent conducts a curve (these trajectories are suppossed to be harder -> Higher ADE and FDE)
                if agent_non_linear_obj.item(): traj_kind_list.append(1) # Curved trajectory
                else: traj_kind_list.append(0) # The agent conducts a curve

            pred_traj_fake_rel = pred_traj_fake_rel.unsqueeze(dim=0)

            # Get predictions in absolute coordinates 
            # (relative displacements -> absolute coordinates (around 0,0) -> map coordinates)

            last_obs = obs_traj[-1,agent_idx,:]
            pred_traj_fake = dataset_utils.relative_to_abs_multimodal(pred_traj_fake_rel,last_obs) + map_origin
                                                     
            # Store predictions in absolute (map) coordinates

            batch_size, pred_len, num_modes, xy = pred_traj_fake.shape
            pred_traj_fake = pred_traj_fake.view(-1, pred_len, xy) #  (assuming batch_size = 1) num_modes x pred_len x 2 (x,y)

            key = num_seq[0].cpu().item()

            output_all[key] = pred_traj_fake.cpu().numpy()
            file_list.remove(key)

            end = time.time()
            aux_time += (end-start)
            time_per_iteration = aux_time/(batch_index+1)

            print(f"Time per iteration: {time_per_iteration} s. \n \
                    Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

        # Add sequences not loaded in dataset

        file_list = []
        try:
            print("No processed files: {}".format(len(file_list.keys())))
        except:
            print("All files have been processed")

        for key in file_list:
            output_all[key] = np.zeros((num_modes, 30, 2))

    return output_all, ade_list, fde_list, traj_kind_list, num_seq_list   

def main(args):
    """
    """

    # Load config file

    model = args.model_path.split('/')[2]
    config_path = os.path.join(BASE_DIR,f"config/config_{model}.yml")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        print(yaml.dump(config, default_flow_style=False))
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR
        config.device_gpu = args.device_gpu

    config.dataset.split = "val" # test 
    config.dataset.split_percentage = 1.0 # To generate the final results, must be 1
    config.dataset.batch_size = 1 # Better to build the h5 results file
    config.dataset.num_workers = 0
    config.dataset.class_balance = -1.0 # Do not consider class balance in the split test
    config.dataset.shuffle = False
    if config.dataset.split == "test": # In test, 0 (we do not have the gt). Otherwise, 30
        config.hyperparameters.pred_len = 0 
    else:
        config.hyperparameters.pred_len = 30

    current_cuda = torch.device(f"cuda:{config.device_gpu}")

    # Dataloader

    print(f"Load {config.dataset.split} split...")
    data_split = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                   root_folder=config.dataset.path,
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split=config.dataset.split,
                                                   split_percentage=config.dataset.split_percentage,
                                                   batch_size=config.dataset.batch_size,
                                                   class_balance=config.dataset.class_balance,
                                                   obs_origin=config.hyperparameters.obs_origin,
                                                   data_augmentation=config.dataset.data_augmentation,
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

    exp_name = os.path.join(*args.model_path.split('/')[-3:-1]) # model_type/exp_name (e.g. social_lstm_mhsa/exp1)

    print(f"Evaluate model in {config.dataset.split} split")
    output_all, ade_list, fde_list, traj_kind_list, num_seq_list = \
        evaluate(split_loader, generator, args.num_modes, 
                 config.dataset.split, current_cuda, config.hyperparameters.pred_len)

    ## Create results folder if does not exist

    results_path = os.path.join(config.base_dir,"results",exp_name,config.dataset.split)

    if not os.path.exists(results_path):
        print("Create results path folder: ", results_path)
        os.makedirs(results_path) # os.makedirs create intermediate directories. os.mkdir only the last one

    if config.dataset.split == "test":
        # Generate H5 file for Argoverse Motion-Forecasting competition (only test split)   

        generate_forecasting_h5(output_all, results_path)

    else: # val or train
        # Save ADE and FDE values in CSVç

        mean_ade = round(sum(ade_list) / (len(ade_list)),3)
        mean_fde = round(sum(fde_list) / (len(fde_list)),3)

        # Write ade and fde values in CSV

        with open(results_path+'/metrics.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            header = ['Index','Sequence.csv','Agent_Traj_Kind','ADE','FDE']
            csv_writer.writerow(header)

            sorted_indeces = np.argsort(ade_list)

            for _,sorted_index in enumerate(sorted_indeces):
                traj_kind = traj_kind_list[sorted_index]
                seq_id = num_seq_list[sorted_index].item() 
                curr_ade = round(ade_list[sorted_index],3)
                curr_fde = round(fde_list[sorted_index],3)

                data = [str(sorted_index),str(seq_id),traj_kind,curr_ade,curr_fde]

                csv_writer.writerow(data)

            # Write mean ADE and FDE 

            csv_writer.writerow(['-','-','-'])
            csv_writer.writerow(['Mean',mean_ade,mean_fde])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
python evaluate/argoverse/generate_results.py \
--model_path "save/argoverse/social_lstm_mhsa/best_unimodal_100_percent/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--num_modes 6 --device_gpu 0
"""