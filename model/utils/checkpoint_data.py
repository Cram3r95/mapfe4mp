#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Load checkpoint and generator

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import os
import importlib

from collections import defaultdict

# DL & Math imports

import torch

#######################################

class Checkpoint():
    """
    """
    def __init__(self):
        self.config_cp = {
            "G_losses" : defaultdict(list),
            "D_losses" : defaultdict(list),
            "losses_ts" : [],
            "metrics_val" : defaultdict(list),
            "metrics_train" : defaultdict(list),
            "sample_ts" : [],
            "restore_ts" : [],
            "norm_g" : [],
            "norm_d" : [],
            "counters" : {
                "t": None,
                "epoch": None,
            },
            "g_state" : None,
            "g_optim_state" : None,
            "d_state" : None,
            "d_optim_state" : None,
            "g_best_state" : None,
            "d_best_state" : None,
            "best_t" : None,
            "g_best_nl_state" : None,
            "d_best_state_nl" : None,
            "best_t_nl" : None
        }

    def load_checkpoint(self, config):
        self.config_cp["args"] = config.__dict__
        self.config_cp["G_losses"] = config.G_losses
        self.config_cp["D_losses"] = config.D_losses
        self.config_cp["losses_ts"] = config.losses_ts
        self.config_cp["metrics_val"] = config.metrics_val
        self.config_cp["metrics_train"] = config.metrics_train
        self.config_cp["sample_ts"] = config.sample_ts
        self.config_cp["restore_ts"] = config.restore_ts
        self.config_cp["norm_g"] = config.norm_g
        self.config_cp["norm_d"] = config.norm_d
        self.config_cp["counters"] = config.counters
        self.config_cp["g_state"] = config.g_state
        self.config_cp["g_optim_state"] = config.g_optim_state
        self.config_cp["d_state"] = config.d_state
        self.config_cp["d_optim_state"] = config.d_optim_state
        self.config_cp["g_best_state"] = config.g_best_state
        self.config_cp["d_best_state"] = config.d_best_state
        self.config_cp["best_t"] = config.best_t
        self.config_cp["g_best_nl_state"] = config.g_best_nl_state
        self.config_cp["d_best_state_nl"] = config.d_best_state_nl
        self.config_cp["best_t_nl"] = config.best_t_nl

def get_generator(model_path, config):
    """
    """

    current_cuda = torch.device(f"cuda:{config.device_gpu}")
    device = torch.device(current_cuda if torch.cuda.is_available() else "cpu")

    curr_model = config.model.name
    # adversarial_training = False
    # if "gan" in curr_model: 
    #     curr_model = '_'.join(curr_model.split('_')[1:])
    #     adversarial_training = True
    
    # Find model in the same folder where the weights were stored
    
    filename = os.path.join(config.hyperparameters.output_dir,curr_model+".py")
    if os.path.isfile(filename):
        aux = config.hyperparameters.output_dir.replace("/",".")
        curr_model = f"{aux}.{curr_model}"

    # Otherwise, use current model
    
    else: 
        curr_model = f"model.models.{curr_model}"
   
    curr_model_module = importlib.import_module(curr_model)
    TrajectoryGenerator = getattr(curr_model_module,"TrajectoryGenerator")

    # TODO: Pass as argument only "config"

    generator = TrajectoryGenerator(PHYSICAL_CONTEXT=config.hyperparameters.physical_context,
                                    CURRENT_DEVICE=device)

    checkpoint = torch.load(model_path, map_location=current_cuda)
    generator.load_state_dict(checkpoint.config_cp['g_best_state'], strict=False)
    generator.to(device)
    generator.eval() # We do not want to train during inference

    return generator
 
def get_total_norm(parameters, norm_type=2):
    """
    """
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm