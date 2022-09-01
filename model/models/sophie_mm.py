#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Multimodal model

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb

# DL & Math imports

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom imports

from model.modules.attention import MultiHeadAttention
from model.modules.layers import Linear, LinearRes

# Constants

NUM_MODES = 6

DATA_DIM = 2 # (x,y)
OBS_LEN = 20
PRED_LEN = 30

ENCODER_BIDIRECTIONAL = True
ENCODER_NUM_LAYERS = 2
MODEL_DYNAMICS = True
CONV_FILTERS = 32

NUM_HEADS = 4
DROPOUT = 0.5

DOUBLE_BRANCH_DECODER = True
DECODER_BIDIRECTIONAL = True
DECODER_NUM_LAYERS = 2
H_DIM = 128
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
ACTIVATION_FUNCTION_MLP = "SELU" # ReLU, Tanh, GELU

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if ACTIVATION_FUNCTION_MLP == "ReLU":
            layers.append(nn.ReLU())
        elif ACTIVATION_FUNCTION_MLP == "Tanh":
            layers.append(nn.Tanh())
        elif ACTIVATION_FUNCTION_MLP == "GELU":
            layers.append(nn.GELU())
        elif ACTIVATION_FUNCTION_MLP == "SELU":
            layers.append(nn.SELU())
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.data_dim = DATA_DIM
        self.model_dynamics = MODEL_DYNAMICS
        self.conv_filters = CONV_FILTERS
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM

        if ENCODER_BIDIRECTIONAL: self.D = 2
        else: self.D = 1
        self.num_layers = ENCODER_NUM_LAYERS

        self.SELU = nn.SELU()

        if self.model_dynamics:
            self.conv1 = nn.Conv1d(self.data_dim,self.conv_filters,kernel_size=3, # To model velocities
                                   padding=1,padding_mode="reflect")
            self.conv2 = nn.Conv1d(self.conv_filters,self.conv_filters,kernel_size=3, # To model accelerations
                                   padding=1,padding_mode="reflect")
            self.spatial_embedding = nn.Linear(self.conv_filters*2+self.data_dim, self.bottleneck_dim)
            self.encoder = nn.LSTM(self.bottleneck_dim, self.h_dim, dropout=DROPOUT, 
                                   num_layers=self.num_layers, bidirectional=ENCODER_BIDIRECTIONAL)
        else:
            self.spatial_embedding = nn.Linear(self.data_dim, self.embedding_dim)
            self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, dropout=DROPOUT,
                                   num_layers=self.num_layers, bidirectional=ENCODER_BIDIRECTIONAL)

    def init_hidden(self, batch, current_cuda):
        h = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(current_cuda)
        c = torch.zeros(self.D*self.num_layers, batch, self.h_dim).cuda(current_cuda)
        return h, c

    def forward(self, obs_traj):
        num_agents = obs_traj.size(1)
        state = self.init_hidden(num_agents, current_cuda=obs_traj.device)

        if self.model_dynamics:
            # conv_vel = F.tanh(self.conv1(obs_traj.permute(1,2,0)))
            # conv_acc = F.tanh(self.conv2(conv_vel))
            conv_vel = self.SELU(self.conv1(obs_traj.permute(1,2,0)))
            conv_acc = self.SELU(self.conv2(conv_vel))

            kinematic_state = torch.cat([obs_traj,
                                         conv_vel.permute(2,0,1),
                                         conv_acc.permute(2,0,1)],dim=2) # obs_len x num_agents x (conv_filters·2 + embedding_dim)
            kinematic_state_embedding = self.spatial_embedding(kinematic_state)
            output, state = self.encoder(kinematic_state_embedding, state)
        else:
            obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
            obs_traj_embedding = obs_traj_embedding.view(-1, num_agents, self.embedding_dim)
            output, state = self.encoder(obs_traj_embedding, state)

        if self.D == 2: # LSTM bidirectional
            final_h = state[0][-2,:,:] # Take the forward information from the last stacked LSTM layer
                                       # state, not the reverse

                                       # L0(F->R) -> L1(F->R) -> L2(F->R) ...
        else:
            final_h = state[0][-1,:,:] # Take the information from the last LSTM layer

        final_h = final_h.view(num_agents, self.h_dim)

        return final_h

class MMDecoderDoubleBranch(nn.Module):
    def __init__(self):
        super(MMDecoderDoubleBranch, self).__init__()
        self.data_dim = DATA_DIM
        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.num_modes = NUM_MODES

        self.SELU = nn.SELU()
        norm = "GN"
        ng = 1

        traj_points = self.num_modes*self.data_dim
        self.spatial_embedding = nn.Linear(traj_points, self.embedding_dim) 
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)

        # Head

        pred = []
        for i in range(self.num_modes):
            pred.append(
                nn.Sequential(
                    LinearRes(self.h_dim, self.h_dim,norm=norm,ng=ng),
                    nn.Linear(self.h_dim, self.data_dim)
                )
            ) 
        self.hidden2pos = nn.ModuleList(pred) 
        self.confidences = nn.Sequential(
                                         LinearRes(self.h_dim, self.h_dim, norm=norm, ng=ng), 
                                         nn.Linear(self.h_dim, self.num_modes)
                                        ) 

    def forward(self, last_obs, last_obs_rel, state_tuple):
        """
        last_obs (batch_size, self.data_dim)
        last_obs_rel (batch_size, self.data_dim)
        state_tuple: h and c
            h : c : (batch_size, self.h_dim)
        """

        batch_size, data_dim = last_obs.shape
        assert data_dim == self.data_dim

        last_obs_rel = last_obs_rel.view(1,batch_size,1,data_dim)
        last_obs_rel = last_obs_rel.repeat_interleave(self.num_modes, dim=2)
        last_obs_rel = last_obs_rel.view(1, batch_size, -1)

        pred_traj_fake_rel = []
        decoder_input = self.SELU(self.spatial_embedding(last_obs_rel.contiguous().view(batch_size, -1))) 
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)

        for _ in range(self.pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) 

            rel_pos = []
            for num_mode in range(self.num_modes):
                rel_pos_ = self.hidden2pos[num_mode](state_tuple[0].contiguous().view(-1, self.h_dim))
                rel_pos.append(rel_pos_)
            rel_pos = torch.cat(rel_pos,dim=1)
            
            decoder_input = self.SELU(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

        # Multimodal prediction in relative displacements

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim
        
        # Confidences

        conf = self.confidences(state_tuple[0].contiguous().view(-1, self.h_dim))
        conf = torch.softmax(conf, dim=1) # batch_size, num_modes

        return pred_traj_fake_rel, conf

class MMDecoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.data_dim = DATA_DIM
        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.num_modes = NUM_MODES
   
        traj_points = self.num_modes*self.data_dim
        self.spatial_embedding = nn.Linear(traj_points, self.embedding_dim)   
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        
        # Head

        self.hidden2pos = nn.Linear(self.h_dim, traj_points)
        self.confidences = nn.Linear(self.h_dim, self.num_modes)

    def forward(self, last_obs, last_obs_rel, state_tuple):
        """
        last_obs (batch_size, self.data_dim)
        last_obs_rel (batch_size, self.data_dim)
        state_tuple: h and c
            h : c : (batch_size, self.h_dim)
        """

        batch_size, data_dim = last_obs.shape
        assert data_dim == self.data_dim

        last_obs_rel = last_obs_rel.view(1,batch_size,1,data_dim)
        last_obs_rel = last_obs_rel.repeat_interleave(self.num_modes, dim=2)
        last_obs_rel = last_obs_rel.view(1, batch_size, -1)

        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(last_obs_rel.contiguous().view(batch_size, -1))) 
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)

        for _ in range(self.pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) 

            rel_pos = self.hidden2pos(state_tuple[0].contiguous().view(-1, self.h_dim))

            decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

        # Multimodal prediction in relative displacements

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim
        
        # Confidences

        conf = self.confidences(state_tuple[0].contiguous().view(-1, self.h_dim))
        conf = torch.softmax(conf, dim=1) # batch_size, num_modes

        return pred_traj_fake_rel, conf

class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM

        # Encode information

        self.encoder = Encoder()

        self.sattn = MultiHeadAttention(key_size=self.h_dim, 
                                        query_size=self.h_dim, 
                                        value_size=self.h_dim,
                                        num_hiddens=self.h_dim, 
                                        num_heads=NUM_HEADS, 
                                        dropout=DROPOUT)

        # Decode multimodal prediction
         
        mlp_context_input_dim = self.h_dim*2 # Encoded trajectories + social context = config_encoder_lstm.h_dim*2 # Encoded trajectories + social context
        self.mlp_dim = round((mlp_context_input_dim + self.h_dim) / 2)
        mlp_decoder_context_dims = [mlp_context_input_dim, self.mlp_dim, self.h_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)

        if DOUBLE_BRANCH_DECODER:
            self.decoder = MMDecoderDoubleBranch()
        else:
            self.decoder = MMDecoderLSTM()
        

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx):
        current_cuda = obs_traj.device
        final_encoder_h = self.encoder(obs_traj_rel)

        batch_features = []

        for start, end in seq_start_end.data: # Iterate per sequence
            curr_final_encoder_h = final_encoder_h[start:end,:] # seq x DATA_DIM
            curr_final_encoder_h = torch.unsqueeze(curr_final_encoder_h, 0) # Required by Attention

            curr_social_attn = self.sattn(
                                          curr_final_encoder_h, # Key
                                          curr_final_encoder_h, # Query
                                          curr_final_encoder_h, # Value
                                          None
                                         )

            concat_features = torch.cat(
                        [
                            curr_final_encoder_h.contiguous().view(-1,self.h_dim),
                            curr_social_attn.contiguous().view(-1,self.h_dim)
                        ],
                        dim=1
                    )

            batch_features.append(concat_features)

        mlp_decoder_context_input = torch.cat(batch_features,axis=0) # batch_size · num_agents x num_inputs_decoder · hidden_dim

        # Get agent last observations (both abs (around 0,0) and rel-rel)

        if agent_idx is not None: # for single agent prediction
            last_pos = obs_traj[:, agent_idx, :]
            last_pos_rel = obs_traj_rel[:, agent_idx, :]

            mlp_decoder_context_input = mlp_decoder_context_input[agent_idx,:]
            decoder_h = self.mlp_decoder_context(mlp_decoder_context_input).unsqueeze(0)

            decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(current_cuda)
            state_tuple = (decoder_h, decoder_c)

            last_pos = obs_traj[-1, agent_idx, :]
            last_pos_rel = obs_traj_rel[-1, agent_idx, :]
        else:
            decoder_h = self.mlp_decoder_context(mlp_decoder_context_input)
            decoder_h = torch.unsqueeze(decoder_h, 0)

            decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(current_cuda)
            state_tuple = (decoder_h, decoder_c)

            last_pos = obs_traj[-1, :, :]
            last_pos_rel = obs_traj_rel[-1, :, :]

        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel