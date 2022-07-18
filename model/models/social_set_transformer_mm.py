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
    def __init__(self, dim_input=20*2, num_outputs=3, dim_output=30 * 2,
                 num_inds=8, dim_hidden=64, num_heads=4, ln=False, pred_len=30):
        """
        Num outputs = Multimodality
        """

        super(TrajectoryGenerator, self).__init__()

        self.num_outputs = num_outputs
        self.pred_len = pred_len
        
        # Layers

        self.emb = nn.Linear(2, dim_hidden) # Embedding (from x|y to dim_hidden)
        self.enc = nn.Sequential( # Encoder
                                 ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                                 ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                                ) 
        self.dec = nn.Sequential( # Decoder
                                 PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                                 SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
                                ) 
        self.regressor = nn.Linear(dim_hidden, dim_output)
        self.mode_confidences = nn.Linear(dim_hidden, 1)

    def forward(self, seq_list_rel, seq_start_end): #
        """
        seq_list_rel: Observation data (20 = obs_len x batch_size·num_agents x data_dimensionality = 2 (x|y))
                      Note that this data is in relative coordinates (displacements)
        start_end_seq: batch_size x 2 (each element indicates the number of agents per sequence batch element
                       in seq_list_rel
        """
        # pdb.set_trace()
        XX = []
        for start, end in seq_start_end.data:
            Y = seq_list_rel[:,start:end,:].contiguous().permute(1,0,2)
            num_agents, obs_len, data_dim = Y.shape
            Y = Y.contiguous().view(1, num_agents, data_dim*obs_len)
            # X = self.emb(X)
            Y = self.dec(self.enc(Y)) # 1, m, 64
            # Y = Y.view(n,t,-1) # (n, obs, h_dim)
            XX.append(Y)
        XX = torch.cat(XX, 0)

        coords = self.regressor(XX).reshape(-1, self.num_outputs, self.pred_len, 2) # (b, m, t, 2)
        #pdb.set_trace()
        confidences = torch.squeeze(self.mode_confidences(XX), -1) # (b, m)
        confidences = torch.softmax(confidences, dim=1)

        return coords, confidences