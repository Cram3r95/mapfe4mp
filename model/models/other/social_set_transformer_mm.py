#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Argoverse dataset

"""
Created on Mon May 23 17:54:37 2022
@author: Miguel Eduardo Ortiz Huamaní, Marcos V. Conde and Carlos Gómez-Huélamo
"""

# General purpose imports

import random
import os
import time
from operator import itemgetter
import pdb

# DL & Math imports

import numpy as np
import torch
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import pdb

# Custom imports

from model.modules.set_transformer import ISAB, PMA, SAB

#######################################

# Multimodal Set transformer

# Kaggle challenge: https://gdude.de/blog/2021-02-05/Kaggle-Lyft-solution
# Code for Lyft: https://github.com/asanakoy/kaggle-lyft-motion-prediction-av
# Read: Set transformer 
# Read: https://medium.com/wovenplanetlevel5/how-to-build-a-motion-prediction-model-for-autonomous-vehicles-29f7f81f1580

class TrajectoryGenerator(nn.Module):
    def __init__(self, dim_input=20*2, num_modes=6, dim_output=30 * 2,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, pred_len=30,
            PHYSICAL_CONTEXT="social", CURRENT_DEVICE="cpu"):
        super(TrajectoryGenerator, self).__init__()
        
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.data_dim = 2
        
        self.enc = nn.Sequential( # Encoder
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential( # Decoder
                PMA(dim_hidden, num_heads, num_modes, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))

        # Head
        
        self.regressor = nn.Linear(dim_hidden, dim_output)
        self.mode_confidences = nn.Linear(dim_hidden, 1)
        
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=None, relevant_centerlines=None):
        """_summary_

        Args:
            obs_traj (_type_): _description_
            obs_traj_rel (_type_): _description_
            seq_start_end (_type_): _description_
            agent_idx (_type_): _description_
            phy_info (_type_, optional): _description_. Defaults to None.
            relevant_centerlines (_type_, optional): _description_. Defaults to None.
        """
        
        batch_size = seq_start_end.shape[0]
        social_features_list = []
        
        for start, end in seq_start_end.data:
            # Get current social information
            
            curr_obs_traj_rel = obs_traj_rel[:,start:end,:].contiguous().permute(1,0,2) # agents x obs_len x data_dim
            num_agents, obs_len, data_dim = curr_obs_traj_rel.shape
            curr_obs_traj_rel = curr_obs_traj_rel.contiguous().view(1, num_agents, data_dim*obs_len)
            
            # Auto-encode current social information
            pdb.set_trace()
            social_features = self.dec(self.enc(curr_obs_traj_rel)) # 1 x num_modes x hidden_dim
            social_features_list.append(social_features)
            
        # Concat social latent space along batch dimension
        
        social_features = torch.cat(social_features_list,0)
        
        # Get Multi-modal prediction and confidences 
        # Here we assume the social latent space is around the target agent
        
        pred_traj_fake_rel = self.regressor(social_features).reshape(-1, 
                                                                     self.num_modes,
                                                                     self.pred_len,
                                                                     self.data_dim)
        conf = torch.squeeze(self.mode_confidences(social_features), -1)
        conf = torch.softmax(conf, dim=1)
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
                pdb.set_trace()
    
        return pred_traj_fake_rel, conf