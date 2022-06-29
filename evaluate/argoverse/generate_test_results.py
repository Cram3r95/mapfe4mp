#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Generate results for the test split (Leaderboard)

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import argparse
import json
import os
import sys
import yaml
import pdb
import glob

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
from model.datasets.argoverse.dataset_utils import relative_to_abs_multimodal
from model.utils.checkpoint_data import get_generator

from argoverse.evaluation.competition_util import generate_forecasting_h5

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--num_modes', required=True, default=6, type=int)
parser.add_argument("--device_gpu", required=True, default=0, type=int)

dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]
DEBUG_TEST_RESULTS = True

def evaluate(loader, generator, num_modes, split, current_cuda):
    """
    """

    assert split == "test" and loader.batch_size == 1

    output_all = {}
    test_folder = BASE_DIR+f"/data/datasets/argoverse/motion-forecasting/{split}/data/"
    file_list = glob.glob(os.path.join(test_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    print("file_list ", len(file_list))

    with torch.no_grad(): # When testing, gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print(f"Evaluating batch {batch_index+1}/{len(loader)}")

            # Load batch in device

            batch = [tensor.cuda(current_cuda) for tensor in batch]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, map_origin, num_seq, norm) = batch

            ## Get AGENT (most interesting obstacle) id

            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # Get predictions (relative displacements)

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx)

            ## (Optional) Plot results

            if DEBUG_TEST_RESULTS:
                seq_id = num_seq.item()
                curr_map_origin = map_origin[0]
                curr_first_obs = obs_traj[0,:,:]
                curr_traj_rel = obs_traj_rel
                curr_object_class_id_list = object_cls

                filename = f"data/datasets/argoverse/motion-forecasting/{split}/data_images/{seq_id}.png"

                plot_functions.plot_trajectories(filename,curr_traj_rel,curr_first_obs,
                                                     curr_map_origin,curr_object_class_id_list,dist_rasterized_map,
                                                     rot_angle=-1,obs_len=obs_traj.shape[0],
                                                     smoothen=False, save=True)#,pred_trajectories_rel=pred_traj_fake_rel)

            ## TODO: Avoid repeating the prediction -> Use a multimodal generator!
            pdb.set_trace()
            pred_traj_fake_rel = torch.repeat_interleave(pred_traj_fake_rel,num_modes,dim=1) # 30,1,2 -> 30,6,2

            # Get predictions in absolute coordinates 
            # (relative displacements -> absolute coordinates (around 0,0) -> map coordinates)

            first_obs = obs_traj[-1,agent_idx,:]
            pred_traj_fake = relative_to_abs_multimodal(pred_traj_fake_rel,first_obs) + \
                             map_origin

            # Store predictions in absolute (map) coordinates

            batch_size, pred_len, modes, xy = pred_traj_fake.shape
            pred_traj_fake = pred_traj_fake.view(-1, pred_len, xy) # num_modes x pred_len x 2 (x,y)

            key = num_seq[0].cpu().item()

            output_all[key] = pred_traj_fake.cpu().numpy()
            file_list.remove(key)

        # Add sequences not loaded in dataset

        file_list = []
        try:
            print("No processed files: {}".format(len(file_list.keys())))
        except:
            print("All files have been processed")

        for key in file_list:
            output_all[key] = np.zeros((num_modes, 30, 2))

    return output_all     

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

    config.dataset.split = "test"
    config.dataset.split_percentage = 1.0 # To generate the final results, must be 1 (whole split test)
    config.dataset.batch_size = 1 # Better to build the h5 results file
    config.dataset.num_workers = 0
    config.dataset.class_balance = -1.0 # Do not consider class balance in the split test
    config.dataset.shuffle = False
    config.hyperparameters.pred_len = 0 # In test, we do not have the gt (prediction points)

    current_cuda = torch.device(f"cuda:{config.device_gpu}")

    # Dataloader

    print("Load test split...")
    data_test = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
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

    test_loader = DataLoader(data_test,
                              batch_size=config.dataset.batch_size,
                              shuffle=config.dataset.shuffle,
                              num_workers=config.dataset.num_workers,
                              collate_fn=seq_collate)

    # Get generator

    print("Load generator...")
    generator = get_generator(args.model_path, config)

    # Evaluate and get metrics

    ## Get experiment name

    exp_name = args.model_path.split('/')[-2]

    ## Create results folder if does not exist

    file_dir = str(Path(__file__).resolve().parent)
    results_path = file_dir+"/results/"+exp_name

    if not os.path.exists(results_path):
        print("Create results path folder: ", results_path)
        os.makedirs(results_path) # os.makedirs create intermediate directories. os.mkdir only the last one

    print("Evaluate test split")
    output_all = evaluate(test_loader, generator, args.num_modes, 
                          config.dataset.split, current_cuda)

    # Generate H5 file for Argoverse Motion-Forecasting competition

    generate_forecasting_h5(output_all, results_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# python evaluate/argoverse/generate_test_results.py --model_path 
# "save/argoverse/social_lstm_mhsa/best_unimodal_100_percent/argoverse_motion_forecasting_dataset_0_with_model.pt" 
# --num_modes 6 --device_gpu 0