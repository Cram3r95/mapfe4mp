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
from prodict import Prodict
from datetime import datetime
from pathlib import Path
import logging
import os
import sys
import argparse
import importlib
import pdb

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

TRAINER_LIST = ["social_lstm_mhsa",
                "social_set_transformer_goals_mm", "social_set_transformer_mm"]

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
    parser.add_argument("--device_gpu", required=True, type=int)
    parser.add_argument("--from_ckpt", type=str, default=None)
    parser.add_argument("--num_ckpt", type=str, default="0")
    args = parser.parse_args()
    print(args.trainer)
    
    # Get model trainer for the current architecture

    curr_trainer = "model.trainers.trainer_%s" % args.trainer
    curr_trainer_module = importlib.import_module(curr_trainer)
    model_trainer = curr_trainer_module.model_trainer

    # Get configuration for the current architecture

    if not args.from_ckpt: # Initialize new experiment from your current config file in configs folder
        config_path = "./config/config_%s.yml" % args.trainer
    elif os.path.isdir(args.from_ckpt): # Continue training from previous checkpoint with the 
                                        # corresponding config file
        config_path = os.path.join(args.from_ckpt,"config_file.yml")
    else:
        assert 1 == 0, "Checkpoint not found!" 
        
    print("BASE_DIR: ", BASE_DIR)

    with open(config_path) as config_file:
        config_file = yaml.safe_load(config_file)
        config_file["device_gpu"] = args.device_gpu

        config_file["base_dir"] = BASE_DIR
        exp_path = os.path.join(config_file["base_dir"], config_file["hyperparameters"]["output_dir"])   
        route_path = exp_path + "/config_file.yml"

        if args.from_ckpt and os.path.isdir(args.from_ckpt): # Overwrite checkpoint_start_from
            model = config_file["dataset_name"] + "_" + args.num_ckpt + "_with_model.pt"

            config_file["hyperparameters"]["checkpoint_start_from"] = os.path.join(args.from_ckpt,
                                                                             model)

        if not os.path.exists(exp_path):
            print("Create experiment path: ", exp_path)
            os.makedirs(exp_path) # makedirs creates intermediate folders

        with open(route_path,'w') as yaml_file:
            yaml.dump(config_file, yaml_file, default_flow_style=False)

        config_file = Prodict.from_dict(config_file)

    now = datetime.now()
    time = now.strftime("%H:%M:%S")

    logger = create_logger(os.path.join(exp_path, f"{config_file.dataset_name}_{time}.log"))
    logger.info("Config file: {}".format(config_path))

    model_trainer(config_file, logger)