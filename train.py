#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Main file to train the model

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import yaml
import git
import logging
import os
import sys
import argparse
import importlib
import pdb

from prodict import Prodict
from datetime import datetime

#######################################

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

TRAINER_LIST = [
                "social_lstm_mhsa",
                "social_lstm_mhsa_mm"
                "social_set_transformer",
                "social_set_transformer_mm",
                "gan_social_lstm_mhsa"
               ]

def create_logger(file_path):
    """
    """
    FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=str, choices=TRAINER_LIST)
    parser.add_argument("--device_gpu", required=True, type=int, default=0)
    parser.add_argument("--from_exp", type=str, default=None)
    parser.add_argument("--num_ckpt", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="save")
    args = parser.parse_args()
    print(args.trainer)
    
    # Get model trainer for the current architecture

    curr_trainer = "model.trainers.trainer_%s" % args.trainer
    curr_trainer_module = importlib.import_module(curr_trainer)
    model_trainer = curr_trainer_module.model_trainer

    # Get configuration for the current architecture

    if not args.from_exp: # Initialize new experiment from your current 
                          # config file in configs folder
        config_path = "./config/config_%s.yml" % args.trainer
    else: # Continue training from previous checkpoint with the 
          # corresponding config file
        assert os.path.isdir(args.from_exp), print("Checkpoint not found!")
        config_path = os.path.join(args.from_exp,"config_file.yml")

    now = datetime.now()
    exp_name = now.strftime("exp-%Y-%m-%d_%Hh") # -%M")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
        config["device_gpu"] = args.device_gpu
        config["base_dir"] = BASE_DIR
        
        if not config["hyperparameters"]["exp_name"]: # Empty -> Fill with current hour and day
            config["hyperparameters"]["exp_name"] = exp_name

        split_percentage_str = str(100*config["dataset"]["split_percentage"]) + "_percent" 
        config["hyperparameters"]["output_dir"] = os.path.join(config["hyperparameters"]["save_root_dir"],
                                                               config["model"]["name"],
                                                               split_percentage_str,
                                                               config["hyperparameters"]["exp_name"])

        route_path = config["hyperparameters"]["output_dir"] + "/config_file.yml"
  
        if args.from_exp and os.path.isdir(args.from_exp): # Overwrite checkpoint_start_from
            model = config["dataset_name"] + "_" + args.num_ckpt + "_with_model.pt"

            config["hyperparameters"]["checkpoint_start_from"] = os.path.join(args.from_exp,model)
        else:
            dataset_name = config["dataset_name"]
            filename = os.path.join(config["hyperparameters"]["output_dir"],f"{dataset_name}_0_with_model.pt")
            assert not os.path.exists(filename),print("This path already has a checkpoint!")

            # If the dir does not exist, or it does not have a checkpoint, delete

            path_to_remove = config["hyperparameters"]["output_dir"]
            os.system(f"rm -rf {path_to_remove}")
  
        if not os.path.exists(config["hyperparameters"]["output_dir"]):
            print("Create experiment path: ", config["hyperparameters"]["output_dir"])
            os.makedirs(config["hyperparameters"]["output_dir"]) # makedirs creates intermediate folders

        with open(route_path,'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        config = Prodict.from_dict(config)

    # Create logger

    logger = create_logger(os.path.join(config["hyperparameters"]["output_dir"], f"{config.dataset_name}_{exp_name}.log"))
    logger.info("Config file: {}".format(config_path))

    # Modify some variables of the configuration given input arguments
    # E.g. The original config has batch_size = 1024 but now your GPU is
    # almost full -> You keep 1024 in the file (as a reference) but here
    # you specify a smaller batch_size (e.g. 512), that is, we do not want
    # to incorporate these temporal changes to the original config file

    if args.batch_size: config.dataset.batch_size = args.batch_size
    if args.output_dir != "save": 
        config.hyperparameters.output_dir = args.output_dir

    model_trainer(config, logger)