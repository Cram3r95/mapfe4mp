#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Encoder functions

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo, Miguel Eduardo Ortiz Huamaní and Marcos V. Conde
"""

# General purpose imports

import pdb

# DL & Math imports

import torch
from torch import nn
from prodict import Prodict
import torch.nn.functional as F

# Custom imports

from model.modules.layers import MLP

class EncoderLSTM(nn.Module):
    """
    """
    def __init__(self, embedding_dim=16, h_dim=64, num_layers=1, bidirectional=False, current_cuda="cuda:0"):
        super().__init__()

        self.data_dim = 2 # x,y
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim # Number of LSTM units (per layer)
        self.num_layers = num_layers

        if bidirectional: self.D = 2
        else: self.D = 1

        self.spatial_embedding = nn.Linear(self.data_dim, self.embedding_dim)
        
        # TODO: Spatial embedding required?

        # self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)#, bidirectional=bidirectional)
        # self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, num_layers, bidirectional=bidirectional)

        self.encoder = nn.LSTM(self.data_dim, self.h_dim, num_layers, bidirectional=bidirectional)
        
        self.current_cuda = current_cuda

    def init_hidden(self, batch):
        """
        """
        h = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(self.current_cuda)
        c = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(self.current_cuda)
        return h, c

    def forward(self, obs_traj):
        """
        """
        n_agents = obs_traj.size(1)
        state = self.init_hidden(n_agents)

        # TODO: Spatial embedding required?
        
        # obs_traj_embedding = F.leaky_relu(self.spatial_embedding(obs_traj.contiguous().view(-1, 2)))
        # obs_traj_embedding = obs_traj_embedding.view(-1, n_agents, self.embedding_dim)
        # output, state = self.encoder(obs_traj_embedding, state)
        
        output, state = self.encoder(obs_traj, state)

        if self.D == 2: # LSTM bidirectional
            final_h = state[0][0,:,:] # Take the forward information from the state, not the reverse
        else:
            final_h = state[0]

        final_h = final_h.view(n_agents, self.h_dim)
        
        return final_h

class BaseEncoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError