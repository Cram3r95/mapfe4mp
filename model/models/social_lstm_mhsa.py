#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## LSTM based Encoder-Decoder with Multi-Head Self Attention

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo, Miguel Eduardo Ortiz Huamaní and Marcos V. Conde
"""

# General purpose imports 

from os import path
import pdb
import math

# DL & Math imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Custom imports

from model.modules.encoders import EncoderLSTM as Encoder
from model.modules.attention import MultiHeadAttention
from model.modules.decoders import DecoderLSTM as Decoder
from model.modules.decoders import TemporalDecoderLSTM as TemporalDecoder

# Aux functions

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)

class TrajectoryGenerator(nn.Module):
    def __init__(self, config_encoder_lstm, config_decoder_lstm, config_mhsa, current_cuda="cuda:0"):
        super(TrajectoryGenerator, self).__init__() # initialize nn.Module with default parameters

        # Aux variables

        self.current_cuda = current_cuda
        
        # Encoder

        self.encoder = Encoder(h_dim=config_encoder_lstm.h_dim, # Num units
                               bidirectional=config_encoder_lstm.bidirectional, 
                               num_layers=config_encoder_lstm.num_layers,
                               dropout=config_encoder_lstm.dropout,
                               current_cuda=self.current_cuda)
        self.lne = nn.LayerNorm(config_encoder_lstm.h_dim) # Layer normalization encoder

        # Attention

        self.sattn = MultiHeadAttention(key_size=config_mhsa.h_dim, 
                                        query_size=config_mhsa.h_dim, 
                                        value_size=config_mhsa.h_dim,
                                        num_hiddens=config_mhsa.h_dim, 
                                        num_heads=config_mhsa.num_heads, 
                                        dropout=config_mhsa.dropout)

        ## Final context (Encoded trajectories + Social Attention)

        assert config_mhsa.h_dim == config_encoder_lstm.h_dim
        self.context_h_dim = config_mhsa.h_dim

        mlp_context_input = config_encoder_lstm.h_dim*2 # *2 since we want to concat of social context and encoded trajectories
        self.lnc = nn.LayerNorm(mlp_context_input) # Layer normalization context decoder

        # Decoder

        ## Decoder input context

        assert config_decoder_lstm.h_dim == config_encoder_lstm.h_dim

        #mlp_decoder_context_dims = [mlp_context_input, config_decoder_lstm.mlp_dim, config_decoder_lstm.h_dim]
        mlp_decoder_context_dims = [mlp_context_input, *config_decoder_lstm.mlp_dim, config_decoder_lstm.h_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)

        ## Final prediction
        
        self.decoder = TemporalDecoder(h_dim=config_decoder_lstm.h_dim,
                                       embedding_dim=config_decoder_lstm.embedding_dim,
                                       num_layers=config_decoder_lstm.num_layers,
                                       bidirectional=config_decoder_lstm.bidirectional,
                                       dropout=config_decoder_lstm.dropout,
                                       current_cuda=self.current_cuda)

    def forward(self, obs_traj, obs_traj_rel, start_end_seq, agent_idx=None):
        """
        Assuming obs_len = 20, pred_len = 30:

        n: number of objects in all the scenes of the batch
        b: batch
        obs_traj: (20,n,2)
        obs_traj_rel: (20,n,2)
        start_end_seq: (b,2)
        agent_idx: (b, 1) -> index of AGENT (of interest) in every sequence.
            None: trajectories for every object in the scene will be generated
            Not None: just trajectories for the agent in the scene will be generated
        -----------------------------------------------------------------------------
        pred_traj_fake_rel:
            (30,n,2) -> if agent_idx is None
            (30,b,2)
        """
        
        # Encode trajectory (encode per sequence)

        final_encoder_h = []

        for start, end in start_end_seq.data:
            curr_obs_traj_rel = obs_traj_rel[:,start:end,:]
            curr_final_encoder_h = self.encoder(curr_obs_traj_rel) 
            curr_final_encoder_h = torch.unsqueeze(curr_final_encoder_h, 0)
            final_encoder_h.append(curr_final_encoder_h)

        final_encoder_h = torch.cat(final_encoder_h,1)
        final_encoder_h = self.lne(final_encoder_h)

        # Social Attention to encoded trajectories (pay attention per sequence)

        # TODO: Test with Global MHSA (considering all agents of the scene, not
        # only the AGENT of interest)

        attn_s = []
        for start, end in start_end_seq.data:
            attn_s_batch = self.sattn(final_encoder_h[:,start:end,:], # Key
                                      final_encoder_h[:,start:end,:], # Query
                                      final_encoder_h[:,start:end,:], # Value
                                      None
                                      )
            attn_s.append(attn_s_batch)
        attn_s = torch.cat(attn_s, 1)
        
        # Create decoder context input

        mlp_decoder_context_input = torch.cat(
            [
                final_encoder_h.contiguous().view(-1, self.context_h_dim), 
                attn_s.contiguous().view(-1, self.context_h_dim)
            ],
            dim=1
        ) 

        # Take the encoded trajectory and attention regarding only the AGENT for each sequence
        # if NOT none

        if agent_idx is not None:
            mlp_decoder_context_input = mlp_decoder_context_input[agent_idx,:]

        decoder_h = self.mlp_decoder_context(self.lnc(mlp_decoder_context_input))
        decoder_h = torch.unsqueeze(decoder_h, 0) 

        # decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(self.current_cuda)

        # state_tuple = (decoder_h, decoder_c)

        # Get agent observations

        if agent_idx is not None: # for single agent prediction
            last_pos = obs_traj[:, agent_idx, :]
            last_pos_rel = obs_traj_rel[:, agent_idx, :]
        else:
            last_pos = obs_traj[-1, :, :]
            last_pos_rel = obs_traj_rel[-1, :, :]

        # Decode trajectories

        # pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, decoder_h)
        
        return pred_traj_fake_rel