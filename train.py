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

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

TRAINER_LIST = ["social_lstm_mhsa", "social_transformer", "social_set_transformer_mm"]

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
    args = parser.parse_args()
    print(args.trainer)
    
    # Get model trainer for the current architecture

    curr_trainer = "model.trainers.trainer_%s" % args.trainer
    curr_trainer_module = importlib.import_module(curr_trainer)
    model_trainer = curr_trainer_module.model_trainer

    # Get configuration for the current architecture

    config_path = "./config/config_%s.yml" % args.trainer
    print("BASE_DIR: ", BASE_DIR)

    with open(config_path) as config_file:
        config_file = yaml.safe_load(config_file)

        # Fill some additional dimensions

        config_file["base_dir"] = BASE_DIR
        exp_path = os.path.join(config_file["base_dir"], config_file["hyperparameters"]["output_dir"])   
        route_path = exp_path + "/config_file.yml"

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