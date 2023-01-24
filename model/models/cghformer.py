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
import sys

# DL & Math imports

import torch 
import torch.nn as nn
import torch.nn.functional as F

if str(sys.version_info[0])+"."+str(sys.version_info[1]) >= "3.9": # Python >= 3.9
    from math import gcd
else:
    from fractions import gcd
    
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse

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
    
# class TrajectoryGenerator(nn.Module):
#     def __init__(self, dim_input=20*2, num_modes=6, dim_output=30 * 2,
#             num_inds=32, dim_hidden=128, num_heads=4, ln=False, pred_len=30,
#             PHYSICAL_CONTEXT="social", CURRENT_DEVICE="cpu"):
#         super(TrajectoryGenerator, self).__init__()
        
#         self.num_modes = num_modes
#         self.pred_len = pred_len
#         self.data_dim = 2
        
#         self.enc = nn.Sequential( # Encoder
#                 ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
#                 ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
#         self.dec = nn.Sequential( # Decoder
#                 PMA(dim_hidden, num_heads, num_modes, ln=ln),
#                 SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
#                 SAB(dim_hidden, dim_hidden, num_heads, ln=ln))

#         # Head
        
#         self.regressor = nn.Linear(dim_hidden, dim_output)
#         self.mode_confidences = nn.Linear(dim_hidden, 1)
        
#     def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=None, relevant_centerlines=None):
#         """_summary_

#         Args:
#             obs_traj (_type_): _description_
#             obs_traj_rel (_type_): _description_
#             seq_start_end (_type_): _description_
#             agent_idx (_type_): _description_
#             phy_info (_type_, optional): _description_. Defaults to None.
#             relevant_centerlines (_type_, optional): _description_. Defaults to None.
#         """
        
#         batch_size = seq_start_end.shape[0]
#         social_features_list = []
        
#         for start, end in seq_start_end.data:
#             # Get current social information
            
#             curr_obs_traj_rel = obs_traj_rel[:,start:end,:].contiguous().permute(1,0,2) # agents x obs_len x data_dim
#             num_agents, obs_len, data_dim = curr_obs_traj_rel.shape
#             curr_obs_traj_rel = curr_obs_traj_rel.contiguous().view(1, num_agents, data_dim*obs_len)
            
#             # Auto-encode current social information
#             pdb.set_trace()
#             social_features = self.dec(self.enc(curr_obs_traj_rel)) # 1 x num_modes x hidden_dim
#             social_features_list.append(social_features)
            
#         # Concat social latent space along batch dimension
        
#         social_features = torch.cat(social_features_list,0)
        
#         # Get Multi-modal prediction and confidences 
#         # Here we assume the social latent space is around the target agent
        
#         pred_traj_fake_rel = self.regressor(social_features).reshape(-1, 
#                                                                      self.num_modes,
#                                                                      self.pred_len,
#                                                                      self.data_dim)
#         conf = torch.squeeze(self.mode_confidences(social_features), -1)
#         conf = torch.softmax(conf, dim=1)
#         if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
#                 pdb.set_trace()
    
#         return pred_traj_fake_rel, conf

# Global variables

DATA_DIM = 2
NUM_MODES = 6

OBS_LEN = 20
PRED_LEN = 30
CENTERLINE_LENGTH = 30
NUM_CENTERLINES = 3

NUM_ATTENTION_HEADS = 4
H_DIM = 128

APPLY_DROPOUT = True
DROPOUT = 0.25

HEAD = "MultiLinear" # SingleLinear, MultiLinear, Non-Autoregressive
DECODER_MM_KIND = "Latent" # Latent (if you want to get a multimodal prediction from the same latent space)
                           # Loop (iterate over different latent spaces)
TEMPORAL_DECODER = True
DIST2GOAL = False
WINDOW_SIZE = 20

def make_mlp(dim_list, activation_function="ReLU", batch_norm=False, dropout=0.0, model_output=False):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_function: str, activation function for all layers TODO: Different AF for every layer?
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))

        if model_output and (index == len(dim_list) - 1): # Not apply Activation Function for the last
                                                          # layer if it is the model output
            pass
        else:
            if activation_function == "ReLU":
                layers.append(nn.ReLU())
            elif activation_function == "GELU":
                layers.append(nn.GELU())
            elif activation_function == "Tanh":
                layers.append(nn.Tanh())
            elif activation_function == "LeakyReLU":
                layers.append(nn.LeakyReLU())
                
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))

        index += 1
    return nn.Sequential(*layers)

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: must be (T, B, H)
        :return:
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Smooth_Encoder(nn.Module):
    def __init__(self, h_dim) -> None:
        super(Smooth_Encoder, self).__init__()
        cov_hidden_channel = h_dim // 8

        input_dim = 2

        self.cov1 = nn.Conv1d(input_dim, cov_hidden_channel, kernel_size=3, padding=1)
        self.cov2 = nn.Conv1d(cov_hidden_channel, 2 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov3 = nn.Conv1d(2 * cov_hidden_channel, 4 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov4 = nn.Conv1d(4 * cov_hidden_channel, 8 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov5 = nn.Conv1d(8 * cov_hidden_channel, 16 * cov_hidden_channel, kernel_size=3, padding=1)

        self.linear = nn.Linear(15 * cov_hidden_channel, h_dim)

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(cov_hidden_channel)
        self.norm3 = nn.BatchNorm1d(2 * cov_hidden_channel)
        self.norm4 = nn.BatchNorm1d(4 * cov_hidden_channel)
        self.norm5 = nn.BatchNorm1d(8 * cov_hidden_channel)
        self.norm6 = nn.LayerNorm(h_dim)
    
    def forward(self, x):
        x = self.norm1(x)

        temp1 = self.norm2(F.relu(self.cov1(x)))
        temp2 = self.norm3(F.relu(self.cov2(temp1)))
        temp3 = self.norm4(F.relu(self.cov3(temp2)))
        temp4 = self.norm5(F.relu(self.cov4(temp3)))

        x = torch.cat((temp1, temp2, temp3, temp4), dim=1)

        x = self.linear(x.permute(0, 2, 1))

        return self.norm6(x)

class actor_smooth_decoder(nn.Module):
    def __init__(self):
        super(actor_smooth_decoder, self).__init__()

        self.test1 = nn.Linear(20, 10)
        self.test2 = nn.Linear(10, 1)
        self.test_norm1 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = x.permute(0, 2, 1)
        x = self.test_norm1(F.relu(self.test1(x)))
        x = self.test2(x).squeeze(2) # num_agents x h_dim_social
   
        return x
    
class ActorSubNet(nn.Module):
    def __init__(self, h_dim, depth=None):
        super(ActorSubNet, self).__init__()
        
        self.h_dim = h_dim
        
        if depth is None:
            depth = 2

        self.smooth_traj = Smooth_Encoder(self.h_dim)
        
        self.Attn = nn.ModuleList([nn.MultiheadAttention(self.h_dim, 8, dropout=0.1) for _ in range(depth)])
        self.Norms = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(depth)])
        
        self.final_layer = actor_smooth_decoder()

    def forward(self, agents):
        """_summary_

        Args:
            agents (torch.tensor): _description_ (obs_len x num_agents x data_dim)
            agents_per_sample (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """

        smooth_input = agents.permute(1,2,0) # num_agents x data_dim x obs_len
        hidden_states_batch = self.smooth_traj(smooth_input)

        for layer_index, layer in enumerate(self.Attn):
            temp = hidden_states_batch
            q = k = v = hidden_states_batch.permute(1,0,2)
            hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=None)[0].permute(1,0,2)  
            hidden_states_batch = hidden_states_batch + temp
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
   
        hidden_states_batch = self.final_layer(hidden_states_batch)
        return hidden_states_batch

class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size // 2)
        self.norm = nn.LayerNorm(output_size // 2)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(output_size // 2, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return x

class map_smooth_decoder(nn.Module):
    def __init__(self):
        super(map_smooth_decoder, self).__init__()

        self.test1 = nn.Linear(30, 15)
        self.test2 = nn.Linear(15, 1)
        self.test_norm1 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.test_norm1(F.relu(self.test1(x)))
        x = self.test2(x).squeeze(2)

        return x
    
class MapSubNet(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(MapSubNet, self).__init__()
        if depth is None:
            depth = 2

        input_dim = 2 # Each point of the centerline has two attributes (xy)

        self.MLPs = nn.ModuleList([MLP(input_dim, hidden_size // 8), MLP(hidden_size // 4, hidden_size // 2)])
        self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size // 8, 8, dropout=0.1), nn.MultiheadAttention(hidden_size // 2, 8, dropout=0.1)])
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size // 4), nn.LayerNorm(hidden_size)])

        self.map_decoder = map_smooth_decoder()
        self.final_layer = nn.Linear(NUM_CENTERLINES*H_DIM,H_DIM)
        
    def forward(self, inputs, inputs_mask):
        batch_size = inputs.shape[0] // NUM_CENTERLINES
        
        hidden_states_batch = inputs
        hidden_states_mask = inputs_mask

        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            temp = hidden_states_batch 
            query = key = value = hidden_states_batch.permute(1,0,2)

            hidden_states_batch = layer(query, key, value=value, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)
            hidden_states_batch = torch.cat([hidden_states_batch, temp], dim=2)
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)

        hidden_states_batch = self.map_decoder(hidden_states_batch)
        hidden_states_batch = self.final_layer(hidden_states_batch.view(batch_size,-1))
        
        hidden_states_mask = hidden_states_mask.view(batch_size,-1)
        
        return hidden_states_batch, hidden_states_mask
    
class GNN(nn.Module):
    def __init__(self, h_dim):
        super(GNN, self).__init__()

        self.latent_size = h_dim

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgraph

        offset = 0
        for i in range(len(agents_per_sample)):

            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)
   
            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())

            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list

            edge_index_subgraph = torch.Tensor(np.asarray(edge_index_subgraph) + offset)

            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        
        edge_index = torch.LongTensor(np.column_stack(edge_index))

        return edge_index

    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        
        # goal - origin
        
        edge_attr = data[cols] - data[rows]

        return edge_attr

    def forward(self, gnn_in, centers, agents_per_sample):
        """_summary_

        Args:
            gnn_in (_type_): _description_
            centers (_type_): _description_
            agents_per_sample (_type_): _description_

        Returns:
            _type_: _description_
        """

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr)) 
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr)) 

        return gnn_out

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads, h_dim):
        super(MultiheadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.latent_size = h_dim

        if APPLY_DROPOUT: self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=DROPOUT)
        else: self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa

        max_agents = max(agents_per_sample)

        padded_att_in = torch.zeros((len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)
        mask = torch.arange(max_agents).to(att_in[0].device) < torch.tensor(agents_per_sample).to(att_in[0].device)[:, None]

        padded_att_in[mask] = att_in
        mask_inverted = ~mask
        mask_inverted = mask_inverted.to(att_in.device)

        padded_att_in_swapped = torch.swapaxes(padded_att_in, 0, 1)

        padded_att_in_swapped, _ = self.multihead_attention(
            padded_att_in_swapped, padded_att_in_swapped, padded_att_in_swapped, key_padding_mask=mask_inverted)

        padded_att_in_reswapped = torch.swapaxes(
            padded_att_in_swapped, 0, 1)

        att_out_batch = [x[0:agents_per_sample[i]]
                            for i, x in enumerate(padded_att_in_reswapped)]

        return att_out_batch

class InfoInteraction(nn.Module):
    def __init__(self, hidden_size, head_num=8, dropout=0.1) -> None:
        super(InfoInteraction, self).__init__()

        self.cross_attn = nn.MultiheadAttention(hidden_size, head_num, dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, hidden_size)

    def forward(self, x_padding, y_padding , y_mask):

        cross_attn_output = self.cross_attn(query=x_padding,
                                            key=y_padding,
                                            value=y_padding,
                                            attn_mask=None,
                                            key_padding_mask=None
                                            )[0]

        x_padding = x_padding + self.dropout2(cross_attn_output)
        x_padding = self.norm2(x_padding)

        output = self.linear1(x_padding)
        output = F.relu(output)
        output = self.dropout3(output)
        output = self.linear2(output)

        x_padding = x_padding + self.dropout4(output)
        x_padding = self.norm3(x_padding)

        return x_padding

class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.linear3 = nn.Linear(n_out, n_out)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
            self.norm3 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.norm3(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
    
class Multimodal_Decoder(nn.Module):
    def __init__(self, decoder_h_dim):
        super(Multimodal_Decoder, self).__init__()
        
        norm = "BN"
        ng = 1

        pred = []
        for _ in range(NUM_MODES):
            pred.append(
                nn.Sequential(
                    LinearRes(decoder_h_dim, decoder_h_dim, norm=norm, ng=ng),
                    nn.Linear(decoder_h_dim, PRED_LEN * DATA_DIM),
                )
            )
        self.pred = nn.ModuleList(pred)

        # self.confidences = nn.Sequential(LinearRes(decoder_h_dim, decoder_h_dim, norm=norm, ng=ng), 
        #                                  nn.Linear(decoder_h_dim, 1))
        self.confidences = PredictionNet(PRED_LEN*DATA_DIM,1,norm=False)
        
    def forward(self, latent_space):
        """_summary_

        Args:
            latent_space (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        batch_size = latent_space.shape[0]
        
        pred_traj_fake_rel = []
        
        for num_mode in range(len(self.pred)):
            curr_pred_traj_fake_rel = self.pred[num_mode](latent_space)
            pred_traj_fake_rel.append(curr_pred_traj_fake_rel.contiguous().view(-1,PRED_LEN,DATA_DIM))
        
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim
        
        conf = self.confidences(pred_traj_fake_rel.contiguous().view(batch_size,NUM_MODES,-1)) 
        conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()
            
        return pred_traj_fake_rel, conf
        
class PredictionNet(nn.Module):
    def __init__(self, h_dim, output_dim, norm=True):
        super(PredictionNet, self).__init__()

        self.latent_size = h_dim
        self.output_dim = output_dim
        self.norm = norm

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size)

        self.output_fc = nn.Linear(self.latent_size, self.output_dim)

    def forward(self, prednet_in):
        """_summary_

        Args:
            prednet_in (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Residual layer

        x = self.weight1(prednet_in)
        if self.norm: x = self.norm1(x)
        x = F.relu(x) 
        x = self.weight2(x)
        if self.norm: x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        
        prednet_out = self.output_fc(x)

        return prednet_out
    
class Temporal_Multimodal_Decoder(nn.Module):
    def __init__(self, decoder_h_dim):
        super(Temporal_Multimodal_Decoder, self).__init__()

        self.data_dim = DATA_DIM
        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.window_size = WINDOW_SIZE
        self.centerline_length = CENTERLINE_LENGTH

        self.decoder_h_dim = decoder_h_dim
        self.num_modes = NUM_MODES

        self.spatial_embedding = nn.Linear(self.window_size*2, self.window_size*2)
        
        self.decoder = nn.LSTM(self.window_size*2, 
                               self.decoder_h_dim, 
                               num_layers=1)
  
        pred = []
        for _ in range(self.num_modes):
            pred.append(nn.Linear(self.decoder_h_dim,self.data_dim))
        self.hidden2pos = nn.ModuleList(pred) 

        self.confidences = PredictionNet(PRED_LEN*DATA_DIM,1,norm=False)
        
    def forward(self, traj_abs, traj_rel, state_tuple, num_mode=None, current_centerlines=None):
        """_summary_

        Args:
            traj_abs (_type_): _description_
            traj_rel (_type_): _description_
            state_tuple (_type_): _description_

        Returns:
            _type_: _description_
        """

        obs_len, batch_size, data_dim = traj_abs.shape
        state_tuple_h, state_tuple_c = state_tuple
        
        pred_traj_fake_rel = []
    
        for num_mode in range(self.num_modes):
            traj_rel_ = torch.clone(traj_rel)
            decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel_.permute(1,0,2).contiguous().view(batch_size,-1))) # bs x window_size·2
        
            decoder_input = decoder_input.unsqueeze(0)
            if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
        
            state_tuple_h_ = torch.clone(state_tuple_h)
            state_tuple_c_ = torch.randn(tuple(state_tuple_h_.shape)).cuda(state_tuple_h_.device)
            
            curr_pred_traj_fake_rel = []
            for _ in range(self.pred_len):
                output, (state_tuple_h_, state_tuple_c_) = self.decoder(decoder_input, (state_tuple_h_, state_tuple_c_)) 
                rel_pos = self.hidden2pos[num_mode](output.contiguous().view(-1, self.decoder_h_dim))
                
                traj_rel_ = torch.roll(traj_rel_, -1, dims=(0))
                traj_rel_[-1] = rel_pos

                curr_pred_traj_fake_rel.append(rel_pos)

                decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel_.permute(1,0,2).contiguous().view(batch_size,-1))) # bs x window_size·2
                decoder_input = decoder_input.unsqueeze(0)
                if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
        
            curr_pred_traj_fake_rel = torch.stack(curr_pred_traj_fake_rel,dim=0)
            curr_pred_traj_fake_rel = curr_pred_traj_fake_rel.permute(1,0,2)
            pred_traj_fake_rel.append(curr_pred_traj_fake_rel)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim

        conf = self.confidences(pred_traj_fake_rel.contiguous().view(batch_size,NUM_MODES,-1))
        conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()
        
        return pred_traj_fake_rel, conf
            
class TrajectoryGenerator(nn.Module):
    def __init__(self, PHYSICAL_CONTEXT="social", CURRENT_DEVICE="cpu"):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.data_dim = DATA_DIM
        self.num_modes = NUM_MODES
        self.num_centerlines = NUM_CENTERLINES

        # Encoder

        ## Social 

        self.social_encoder = ActorSubNet(self.h_dim)
        self.agent_gnn = GNN(h_dim=self.h_dim)
        self.sattn = MultiheadSelfAttention(h_dim=self.h_dim,
                                            num_heads=self.num_attention_heads)
        
        ## Physical 
        
        self.physical_encoder = MapSubNet(self.h_dim)

        ## Information interaction
        
        self.info_interaction = InfoInteraction(self.h_dim)
        
        ## Global interaction
        
        ln = True
        num_inds = 8
        
        self.global_interaction = ISAB(self.h_dim, self.h_dim, self.num_attention_heads, num_inds, ln=ln)

        # Decoder
        
        # self.decoder = Temporal_Multimodal_Decoder(self.h_dim)
        self.decoder = Multimodal_Decoder(self.h_dim)
            
    def add_noise(self, input, factor=1):
        """_summary_

        Args:
            input (_type_): _description_
            factor (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """

        noise = factor * torch.randn(input.shape).to(input)
        noisy_input = input + noise
        return noisy_input
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=None, relevant_centerlines=None):
        """_summary_

        Args:
            obs_traj (_type_): _description_
            obs_traj_rel (_type_): _description_
            seq_start_end (_type_): _description_
            agent_idx (_type_): _description_
            phy_info (_type_, optional): _description_. Defaults to None.
            relevant_centerlines (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        # Encoder
        
        ## Actor encoder
        
        centers = obs_traj[-1,:,:] # x,y (abs coordinates)
        agents_per_sample = (seq_start_end[:,1] - seq_start_end[:,0]).cpu().detach().numpy()
        
        agents_features = self.social_encoder(obs_traj_rel)
        out_agent_gnn = self.agent_gnn(agents_features, centers, agents_per_sample)
        out_self_attention = self.sattn(out_agent_gnn, agents_per_sample)
        social_info = torch.cat(out_self_attention,dim=0) # num_agents x hidden_dim_social

        if np.any(agent_idx): # Single agent
            social_info = social_info[agent_idx,:]
        
        ## Map encoder

        _, num_centerlines, points_centerline, data_dim = relevant_centerlines.shape
        relevant_centerlines = relevant_centerlines.view(-1,points_centerline, data_dim)
   
        rows_mask, _ = torch.where(relevant_centerlines[:,:,0] == 0.0)
        relevant_centerlines_mask = torch.zeros([relevant_centerlines.shape[0],points_centerline], dtype=torch.bool).to(obs_traj)
        mask = torch.unique(rows_mask.reshape(1,-1),dim=1)
        relevant_centerlines_mask[mask,:] = True
        
        physical_info, physical_info_mask = self.physical_encoder(relevant_centerlines, relevant_centerlines_mask)
           
        ## Information interaction
     
        crossed_info = self.info_interaction(social_info, physical_info, physical_info_mask)
        
        # Global interaction

        final_latent_space = self.global_interaction(crossed_info.unsqueeze(0)).squeeze(0)
        
        # Decoder
        
        decoder_h = final_latent_space.unsqueeze(0)
        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(obs_traj.device)

        state_tuple = (decoder_h, decoder_c)
        
        traj_agent_abs = obs_traj[-WINDOW_SIZE:, agent_idx, :]
        traj_agent_abs_rel = obs_traj_rel[-WINDOW_SIZE:, agent_idx, :]

        # pred_traj_fake_rel, conf = self.decoder(traj_agent_abs, traj_agent_abs_rel, state_tuple)
        pred_traj_fake_rel, conf = self.decoder(final_latent_space)
        
        return pred_traj_fake_rel, conf