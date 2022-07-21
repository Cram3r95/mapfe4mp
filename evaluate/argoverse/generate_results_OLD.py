#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 09 13:04:56 2021
@author: Miguel Eduardo Ortiz Huamaní and Carlos Gómez-Huélamo
"""

import argparse
import json
import numpy as np
import os
import sys
import torch
import yaml
import pdb
import glob
import git

from pathlib import Path
from prodict import Prodict
from torch.utils.data import DataLoader

from argoverse.evaluation.competition_util import generate_forecasting_h5

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.plot_functions as plot_functions
from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
import model.datasets.argoverse.dataset_utils as dataset_utils
from model.utils.checkpoint_data import get_generator, get_generator_mp_so
from model.trainers.trainer_social_lstm_mhsa import cal_ade, cal_fde

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/argoverse/', type=str)
parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test', type=str)

# Global variables

seqs_without_agent = 0

# Evaluate model functions

def evaluate(loader, generator, num_samples, pred_len, split, results_path):
    """
    """

    output_all = {}
    test_folder = BASE_DIR+"/data/datasets/argoverse/motion-forecasting/test/data/"
    file_list = glob.glob(os.path.join(test_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    print("file_list ", len(file_list))

    with torch.no_grad(): # When testing, gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print(f"Evaluating batch {batch_index+1}/{len(loader)}")

            batch = [tensor.cuda() for tensor in batch]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list,_) = batch

            predicted_traj = [] # Store k trajectories per sequence for the agent
            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
            for _ in range(num_samples):
                # Get predictions
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx) # seq_start_end)

                # Get predictions in absolute coordinates
                pred_traj_fake = dataset_utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1,agent_idx, :]) + ego_origin[0].unsqueeze(0) # ego_origin.permute(1,0,2)# 30,1,2

                predicted_traj.append(pred_traj_fake)

            predicted_traj = torch.stack(predicted_traj, axis=0).view(num_samples,-1,2) # Num_samples x pred_len x 2 (x,y)

            key = num_seq_list[0].cpu().item()

            output_all[key] = predicted_traj.cpu().numpy()
            file_list.remove(key)

        # add sequences not loaded in dataset
        file_list = []
        try:
            print("No processed files: {}".format(len(file_list.keys())))
        except:
            print("All files have been processed")

        for key in file_list:
            output_all[key] = np.zeros((num_samples, 30, 2))

        # Generate H5 file for Argoverse Motion-Forecasting competition

        generate_forecasting_h5(output_all, results_path)

    return output_all

# # Ad-hoc generator for each dataset

# def get_generator(checkpoint, config):
#     """
#     """
#     generator = TrajectoryGenerator()
#     generator.load_state_dict(checkpoint['g_best_state'], strict=False)
#     generator.cuda() # Use GPU
#     generator.eval()
#     return generator      

def main(args):
    """
    """

    # Load config file from this specific model

    model = args.model_path.split('/')[:-1]
    config_path = os.path.join(BASE_DIR,*model,"config_file.yml")

    with open(config_path) as config_file:
        config_file = yaml.safe_load(config_file)
        print(yaml.dump(config_file, default_flow_style=False))
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR
        # config.device_gpu = args.device_gpu

    ## Fill and modify some additional arguments

    past_observations = config_file.hyperparameters.obs_len
    # num_agents_per_obs = config_file.hyperparameters.num_agents_per_obs
    # config_file.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents_per_obs

    config_file.dataset.split = "test"
    config_file.dataset.split_percentage = 1.0 # To generate the final results, must be 1 (whole split test)
    config_file.dataset.batch_size = 1 # Better to build the h5 results file
    config_file.dataset.num_workers = 0
    config_file.dataset.class_balance = -1.0 # Do not consider class balance in the split test
    config_file.dataset.shuffle = False

    config_file.hyperparameters.pred_len = 0 # In test, we do not have the gt (prediction points)

    # Dataloader
    print("Load test split...")
    data_test = ArgoverseMotionForecastingDataset(dataset_name=config_file.dataset_name,
                                                  root_folder=config_file.dataset.path,
                                                  obs_len=config_file.hyperparameters.obs_len,
                                                  pred_len=config_file.hyperparameters.pred_len,
                                                  distance_threshold=config_file.hyperparameters.distance_threshold,
                                                  split=config_file.dataset.split,
                                                  
                                                  split_percentage=config_file.dataset.split_percentage,
                                                  
                                                  batch_size=config_file.dataset.batch_size,
                                                  class_balance=config_file.dataset.class_balance,
                                                  obs_origin=config_file.hyperparameters.obs_origin)

    test_loader = DataLoader(data_test,
                             batch_size=config_file.dataset.batch_size,
                             shuffle=config_file.dataset.shuffle,
                             num_workers=config_file.dataset.num_workers,
                             collate_fn=seq_collate)

    # Get generator
    print("Load generator...")
    # checkpoint = torch.load(args.model_path)
    # generator = get_generator(checkpoint.config_cp, config_file)
    # generator = get_generator_mp_so(args.model_path, config_file) # OLD gan
    generator = get_generator(args.model_path, config_file)

    # Evaluate, store results in .json file and get metrics

    ## Get experiment name

    exp_name = args.model_path.split('/')[-2]

    ## Create results folder if does not exist

    file_dir = str(Path(__file__).resolve().parent)
    results_path = file_dir+"/results/"+exp_name

    if not os.path.exists(results_path):
        print("Create results path folder: ", results_path)
        os.makedirs(results_path) # os.makedirs create intermediate directories. os.mkdir only the last one

    output_all = evaluate(test_loader, generator, args.num_samples, config_file.hyperparameters.pred_len, 
                          config_file.dataset.split, results_path)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)