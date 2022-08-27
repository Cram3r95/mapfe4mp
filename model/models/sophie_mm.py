import os
import math
import random
import numpy as np
import pdb
from model.modules.attention import MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants

NUM_MODES = 6

OBS_LEN = 20
PRED_LEN = 30

MLP_DIM = 64
H_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32

NUM_HEADS = 4
DROPOUT = 0.2

MAX_PEDS = 64

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        # layers.append(nn.ReLU())
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch, current_cuda):
        h = torch.zeros(1, batch, self.h_dim).cuda(current_cuda)
        c = torch.zeros(1, batch, self.h_dim).cuda(current_cuda)
        return (h, c)

    def forward(self, obs_traj):
        padded = len(obs_traj.shape) == 4
        npeds = obs_traj.size(1)
        total = npeds * (MAX_PEDS if padded else 1)

        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, total, self.embedding_dim)
        state = self.init_hidden(total, current_cuda=obs_traj.device)
        output, state = self.encoder(obs_traj_embedding, state)
        final_h = state[0]
        if padded:
            final_h = final_h.view(npeds, MAX_PEDS, self.h_dim)
        else:
            final_h = final_h.view(npeds, self.h_dim)
        return final_h

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim)

        for _ in range(self.pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(npeds, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class MMDecoderLSTM(nn.Module):
    def __init__(self, pred_len=30, h_dim=64, embedding_dim=16, num_modes=3):
        super().__init__()

        self.pred_len = pred_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_modes = num_modes

        traj_points = self.num_modes*2
        self.spatial_embedding = nn.Linear(traj_points, self.embedding_dim)
        
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        
        self.hidden2pos = nn.Linear(self.h_dim, traj_points)
        self.confidences = nn.Linear(self.h_dim, self.num_modes)

    def forward(self, last_obs, last_obs_rel, state_tuple):
        """
        last_obs (1, b, 2)
        last_obs_rel (1, b, 2)
        state_tuple: h and c
            h : c : (1, b, self.h_dim)
        """

        # pdb.set_trace()

        batch_size, data_dim = last_obs.shape
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

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim
        conf = self.confidences(state_tuple[0].contiguous().view(-1, self.h_dim))
        conf = torch.softmax(conf, dim=1) # batch_size, num_modes

        return pred_traj_fake_rel, conf

class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM

        self.encoder = Encoder()

        # self.sattn = SocialAttention()
        self.sattn = MultiHeadAttention(key_size=self.h_dim, 
                                        query_size=self.h_dim, 
                                        value_size=self.h_dim,
                                        num_hiddens=self.h_dim, 
                                        num_heads=NUM_HEADS, 
                                        dropout=DROPOUT)
        # self.decoder = Decoder()
        self.decoder = MMDecoderLSTM(num_modes=NUM_MODES)

        mlp_context_input_dim = self.h_dim*2 # Encoded trajectories + social context = config_encoder_lstm.h_dim*2 # Encoded trajectories + social context
        mlp_decoder_context_dims = [mlp_context_input_dim, self.mlp_dim, self.h_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx):
        current_cuda = obs_traj.device
        final_encoder_h = self.encoder(obs_traj_rel)

        batch_features = []

        for start, end in seq_start_end.data: # Iterate per sequence
            curr_final_encoder_h = final_encoder_h[start:end,:] # seq x 2
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