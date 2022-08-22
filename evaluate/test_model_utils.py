#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Test model utils

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import time

# DL & Math

import torch
import torchstat
from thop import profile, clever_format
from pthflops import count_ops

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

from model.models.social_lstm_mhsa_old import TrajectoryGenerator
from model.utils.utils import load_weights

def test_load_weights():
    m = TrajectoryGenerator()
    print("model ", m)

    w_path = BASE_DIR + "/" + "save/argoverse/gen_exp/social_lstm/argoverse_motion_forecasting_dataset_0_with_model.pt"
    w = torch.load(w_path)
    w_ = load_weights(m, w.config_cp['g_best_state'])
    
    m.load_state_dict(w_)

test_load_weights()