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

#######################################

class EncoderLSTM(nn.Module):
    """
    """
    def __init__(self, embedding_dim=16, h_dim=64, num_layers=1, bidirectional=False, 
                       dropout=0.5, current_cuda="cuda:0", use_rel_disp=False, conv_filters=16):
        super().__init__()

        self.data_dim = 2 # x,y
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim # Number of LSTM units (per layer)
        self.num_layers = num_layers
        self.current_cuda = current_cuda
        self.use_rel_disp = use_rel_disp
        self.conv_filters = conv_filters

        if bidirectional: self.D = 2
        else: self.D = 1

        # TODO: Spatial embedding required?
        # self.spatial_embedding = nn.Linear(self.data_dim, self.embedding_dim)
        
        if self.use_rel_disp:
            self.encoder = nn.LSTM(self.data_dim, self.h_dim, num_layers, 
                                   bidirectional=bidirectional, dropout=dropout)
        else:
            self.conv1 = nn.Conv1d(self.data_dim,self.conv_filters,kernel_size=3, # To model velocities
                               padding=1,padding_mode="reflect")
            self.conv2 = nn.Conv1d(self.conv_filters,self.conv_filters,kernel_size=3, # To model accelerations
                                padding=1,padding_mode="reflect")
            self.encoder = nn.LSTM(self.conv_filters*2+self.data_dim, self.h_dim, num_layers, 
                                   bidirectional=bidirectional, dropout=dropout)

    def init_hidden(self, batch):
        """
        """
        h = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(self.current_cuda)
        c = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(self.current_cuda)
        return h, c

    def forward(self, obs_traj):
        """
        obs_traj may be in absolute coordinates or relative displacements, depending
        on the configuration file
        """
        n_agents = obs_traj.size(1)
        state = self.init_hidden(n_agents)

        # TODO: Spatial embedding required?

        # obs_traj_embedding = F.leaky_relu(self.spatial_embedding(obs_traj.contiguous().view(-1, 2)))
        # obs_traj_embedding = obs_traj_embedding.view(-1, n_agents, self.embedding_dim)
        # output, state = self.encoder(obs_traj_embedding, state)

        if self.use_rel_disp: # Relative displacements
            output, state = self.encoder(obs_traj, state)
        else: # Absolute positions (around 0,0)
            vel_traj = torch.tanh(self.conv1(obs_traj.permute(1,2,0)))
            acc_traj = torch.tanh(self.conv2(vel_traj))
            kinematic_state = torch.cat([obs_traj,
                                         vel_traj.permute(2,0,1),
                                         acc_traj.permute(2,0,1)],dim=2) # 20 x num_agents x (conv_filters·2 + data_dim)
            output, state = self.encoder(kinematic_state, state)

        if self.D == 2: # LSTM bidirectional
            final_h = state[0][-2,:,:] # Take the forward information from the last stacked LSTM layer
                                       # state, not the reverse

                                       # L0(F->R) -> L1(F->R) -> L2(F->R) ...
        else:
            final_h = state[0][-1,:,:] # Take the information from the last LSTM layer

        final_h = final_h.view(n_agents, self.h_dim)
        
        return final_h

class Old_EncoderLSTM(nn.Module):

    def __init__(self, embedding_dim=16, h_dim=64):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        h = torch.zeros(1,batch, self.h_dim).cuda()
        c = torch.zeros(1,batch, self.h_dim).cuda()
        return h, c

    def forward(self, obs_traj):

        npeds = obs_traj.size(1)

        obs_traj_embedding = F.leaky_relu(self.spatial_embedding(obs_traj.contiguous().view(-1, 2)))
        obs_traj_embedding = obs_traj_embedding.view(-1, npeds, self.embedding_dim)
        state = self.init_hidden(npeds)
        output, state = self.encoder(obs_traj_embedding, state)
        final_h = state[0]
        final_h = final_h.view(npeds, self.h_dim)
        return final_h

class BaseEncoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError