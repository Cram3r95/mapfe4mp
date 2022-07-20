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
import git

from pathlib import Path
from prodict import Prodict

# DL & Math imports

import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.plot_functions as plot_functions
from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
import model.datasets.argoverse.dataset_utils as dataset_utils
from model.utils.checkpoint_data import get_generator, get_generator_mp_so
from model.trainers.trainer_social_lstm_mhsa import cal_ade, cal_fde

from argoverse.evaluation.competition_util import generate_forecasting_h5

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--num_modes', required=True, default=6, type=int)
parser.add_argument("--device_gpu", required=True, default=0, type=int)
parser.add_argument("--split", required=True, default="val", type=str)

# N.B. If you cannot appreciate the difference between groundtruth and our prediction, probably
# is because the map does not cover the whole groundtruth because the AGENT is driving at high 
# speed. In this case, increase the dist_around value (e.g. 60, 80, etc.) and in plot_functions
# represent only the predictions, not map+predictions: 
# 
# full_img_cv = cv2.add(img1_bg,img2_fg) -> full_img_cv = img_lanes

dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]
GENERATE_QUALITATIVE_RESULTS = False
COMPUTE_METRICS = True

def generate_csv(results_path,ade_list,fde_list,num_seq_list,traj_kind_list,sort=False):
    """
    If sort = True, sort by ADE
    """

    if sort:
        results_path = os.path.join(results_path,"metrics_sorted_ade.csv")  
    else:
        results_path = os.path.join(results_path,"metrics.csv")  

    mean_ade = round(sum(ade_list) / (len(ade_list)),3)
    mean_fde = round(sum(fde_list) / (len(fde_list)),3)

    with open(results_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        header = ['Index','Sequence.csv','Agent_Traj_Kind','ADE','FDE']
        csv_writer.writerow(header)

        if sort:
            indeces = np.argsort(ade_list)
        else:
            indeces = np.arange(len(ade_list))

        for _,index in enumerate(indeces):
            traj_kind = traj_kind_list[index]
            seq_id = num_seq_list[index] 
            curr_ade = round(ade_list[index],3)
            curr_fde = round(fde_list[index],3)

            data = [str(index),str(seq_id),traj_kind,curr_ade,curr_fde]

            csv_writer.writerow(data)

        # Write mean ADE and FDE 

        csv_writer.writerow(['-','-','-','-','-'])
        csv_writer.writerow(['-','-','Mean',mean_ade,mean_fde])

def evaluate(loader, generator, num_modes, split, current_cuda, pred_len):
    """
    """
    assert loader.batch_size == 1

    output_all = {}
    ade_list = []
    fde_list = []
    traj_kind_list = []
    num_seq_list = []

    ade_, fde_ = None, None

    data_folder = BASE_DIR+f"/data/datasets/argoverse/motion-forecasting/{split}/data/"
    file_list = glob.glob(os.path.join(data_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    print("Data folder: ", data_folder)
    num_files = len(file_list)
    print("Num files ", num_files)

    # Very hard (validation) (batch_index/csv): 4504/4723, 

    # my_seqs = [21866,36005,29100,27919,40897,32509] # Write here the index (assuming batch_size = 1), not the sequence.csv
                                                      # See index and metrics in results/your_model/your_split/metrics.csv

    time_per_iteration = float(0)
    aux_time = float(0)
    limit = -1 # -1 by default to analyze all files of the specified split percentage

    with torch.no_grad(): # During inference (regardless the split), gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            if limit != -1 and (batch_index+1 > limit):
                break

            if limit != -1: files_remaining = limit - (batch_index+1)
            else: files_remaining = num_files - (batch_index+1)

            if 'my_seqs' in locals(): # Analyze some specific sequences
                if batch_index > max(my_seqs):
                    break
                elif batch_index not in my_seqs:
                    continue
            
            start = time.time()

            print(f"Evaluating batch {batch_index+1}/{len(loader)}")
        
            # Load batch in device

            batch = [tensor.cuda(current_cuda) for tensor in batch]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, map_origin, num_seq, norm) = batch

            seq_id = num_seq.cpu().item()

            ## Get AGENT (most interesting obstacle) id

            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # Get predictions (relative displacements)

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx) # pred_len x batch_size x 2

            ## TODO: Avoid repeating the prediction -> Use a multimodal generator!

            # pred_traj_fake_rel = pred_traj_fake_rel.unsqueeze(dim=0) # pred_len x batch_size x 2 -> batch_size x pred_len x num_modes x 2
            # pred_traj_fake_rel = torch.repeat_interleave(pred_traj_fake_rel,num_modes,dim=2) # 1,30,1,2 -> 1,30,6,2
            pred_traj_fake_rel = torch.repeat_interleave(pred_traj_fake_rel,num_modes,dim=1) # 30,1,2 -> 30,6,2

            if COMPUTE_METRICS and split != "test": # Metrics must be in absolute coordinates (either around 0,0 or around map origin)
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
                
                ade_, ade_l_, ade_nl_ = cal_ade(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)

                fde_, fde_l_, fde_nl_ = cal_fde(agent_pred_gt, agent_pred_fake, agent_linear_obj, agent_non_linear_obj, agent_mask)

                ade_ = ade_.item() / pred_len
                fde_ = fde_.item()

                ade_list.append(ade_)
                fde_list.append(fde_)
                num_seq_list.append(seq_id)
                print("ADE, FDE: ", ade_, fde_)

                # If the agent conducts a curve (these trajectories are suppossed to be harder -> Higher ADE and FDE)
                if agent_non_linear_obj.item(): traj_kind_list.append(1) # Curved trajectory
                else: traj_kind_list.append(0) # The agent conducts a curve

            ## (Optional) Plot results

            if GENERATE_QUALITATIVE_RESULTS and seq_id < 150:
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
                                                 smoothen=False,save=True,pred_trajectories_rel=pred_traj_fake_rel,
                                                 ade_metric=ade_,fde_metric=fde_)
            # else:
            #     assert 1 == 0
            pred_traj_fake_rel = pred_traj_fake_rel.unsqueeze(dim=0)

            # Get predictions in absolute coordinates 
            # (relative displacements -> absolute coordinates (around 0,0) -> map coordinates)

            last_obs = obs_traj[-1,agent_idx,:]
            pred_traj_fake = dataset_utils.relative_to_abs_multimodal(pred_traj_fake_rel,last_obs) + map_origin
                                                     
            # Store predictions in absolute (map) coordinates

            batch_size, pred_len, num_modes, xy = pred_traj_fake.shape
            pred_traj_fake = pred_traj_fake.view(-1, pred_len, xy) #  (assuming batch_size = 1) num_modes x pred_len x 2 (x,y)

            output_all[seq_id] = pred_traj_fake.cpu().numpy()
            file_list.remove(seq_id)

            end = time.time()
            aux_time += (end-start)
            time_per_iteration = aux_time/(batch_index+1)

            print(f"Time per iteration: {time_per_iteration} s. \n \
                    Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

        # Add sequences not loaded in dataset

        try:
            print("No processed files: {}".format(len(file_list)))
        except:
            print("All files have been processed")

        for key in file_list:
            output_all[key] = np.zeros((num_modes, 30, 2))

    return output_all, ade_list, fde_list, traj_kind_list, num_seq_list   

def main(args):
    """
    """

    # Load config file from this specific model

    model = args.model_path.split('/')[:-1]
    config_path = os.path.join(BASE_DIR,*model,"config_file.yml")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        print(yaml.dump(config, default_flow_style=False))
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR
        config.device_gpu = args.device_gpu

    config.dataset.split = args.split 
    config.dataset.split_percentage = 1.0 # To generate the final results, must be 1
    config.dataset.batch_size = 1 # Better to build the h5 results file
    config.dataset.num_workers = 0
    config.dataset.class_balance = -1.0 # Do not consider class balance in the split test
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
    # generator = get_generator(args.model_path, config)
    generator = get_generator_mp_so(args.model_path, config) # OLD gan

    # Evaluate model and get metrics

    ## Get experiment name

    exp_name = os.path.join(*args.model_path.split('/')[-4:-1]) # model_type/split_percentage/exp_name 
                                                                # (e.g. social_lstm_mhsa/100.0_percent/exp1)

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

    elif COMPUTE_METRICS: # val or train

        # Write results (ade and fde with the corresponding trajectory type) in CSV

        generate_csv(results_path,ade_list,fde_list,num_seq_list,traj_kind_list)
        generate_csv(results_path,ade_list,fde_list,num_seq_list,traj_kind_list,sort=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
python evaluate/argoverse/generate_results.py \
--model_path "save/argoverse/social_lstm_mhsa/100.0_percent/exp1/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--num_modes 6 --device_gpu 0 --split "test"
"""

"""
python evaluate/argoverse/generate_results.py \
--model_path "save/argoverse/gan_social_lstm_mhsa/10.0_percent/exp-2022-07-20_00-45/argoverse_motion_forecasting_dataset_0_with_model.pt" \
--num_modes 6 --device_gpu 0 --split "test"
"""