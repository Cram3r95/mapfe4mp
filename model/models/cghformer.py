#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## CGHFormer

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import math
import numpy as np
import pdb
import time

# DL & Math imports

import torch 
import torch.nn as nn
import torch.nn.functional as F

#######################################

# Multimodal Set transformer

# Kaggle challenge: https://gdude.de/blog/2021-02-05/Kaggle-Lyft-solution
# Code for Lyft: https://github.com/asanakoy/kaggle-lyft-motion-prediction-av
# Read: Set transformer 
# Read: https://medium.com/@woven_planet/how-to-build-a-motion-prediction-model-for-autonomous-vehicles-6f1faf7c52af

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    
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