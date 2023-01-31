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

# Global variables

DATA_DIM = 2
NUM_MODES = 6

OBS_LEN = 20
PRED_LEN = 30
CENTERLINE_LENGTH = 40
NUM_CENTERLINES = 3

NUM_ATTENTION_HEADS = 4
H_DIM = 128

APPLY_DROPOUT = True
DROPOUT = 0.15

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

def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

class Actor_Smooth_Encoder(nn.Module):
    def __init__(self, h_dim) -> None:
        super(Actor_Smooth_Encoder, self).__init__()
        self.data_dim = 2
        self.h_dim = h_dim
        self.obs_len = OBS_LEN
        
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        
        self.bn0 = nn.BatchNorm1d(self.data_dim)
        
        self.conv1 = nn.Conv1d(self.data_dim,self.h_dim // 8,kernel_size=self.kernel_size,
                               stride=self.stride,padding=self.padding)
        self.bn1 = nn.BatchNorm1d(self.h_dim // 8)

        self.conv2 = nn.Conv1d(self.h_dim // 8,self.h_dim // 4,kernel_size=self.kernel_size,
                               stride=self.stride,padding=self.padding)
        self.bn2 = nn.BatchNorm1d(self.h_dim // 4)

        # Compute agents length after convolution+pooling

        aux_length = self.obs_len
        for key in self._modules.keys():
            if "conv" in key or "pooling" in key:
                aux_length = calculate_output_length(aux_length, self.kernel_size, self.stride, self.padding)

        assert aux_length >= 2

        self.linear = nn.Linear((self.h_dim // 4)*aux_length,self.h_dim)
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): num_agents x data_dim x obs_len

        Returns:
            y (torch.tensor): 
        """
        x = self.bn0(x)

        temp1 = self.bn1(F.relu(self.conv1(x)))
        temp2 = self.bn2(F.relu(self.conv2(temp1)))
        temp2 = temp2.permute(0,2,1) # num_agents x obs_len x h_dim
        temp3 = self.linear(temp2.contiguous().view(x.shape[0],-1))
        temp3 = self.ln(temp3)
        temp3 = F.dropout(temp3, p=DROPOUT, training=self.training)
        
        return temp3

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, h_dim):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.latent_size = h_dim

        if APPLY_DROPOUT: self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=DROPOUT)
        else: self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample, att_mask=None):
        att_out_batch = []

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
    
class ActorSubNet(nn.Module):
    def __init__(self, h_dim):
        super(ActorSubNet, self).__init__()
        
        self.h_dim = h_dim  
        self.num_attention_heads = NUM_ATTENTION_HEADS  
        self.depth = 2

        self.smooth_traj = Actor_Smooth_Encoder(self.h_dim)
            
        self.Attn = nn.ModuleList([MultiHeadSelfAttention(h_dim=self.h_dim, num_heads=self.num_attention_heads) for _ in range(self.depth)])
        self.Norms = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.depth)])

    def forward(self, obs_traj_rel, agents_per_sample):
        """_summary_

        Args:
            agents (torch.tensor): (obs_len x num_agents x data_dim)
            agents (np.array): 
        Returns:
            hidden_states_batch (torch.tensor): num_agents x h_dim -> social agent features 
        """

        # Encode trajectories

        smooth_input = obs_traj_rel.permute(1,2,0) # num_agents x data_dim x obs_len
        hidden_states_batch = self.smooth_traj(smooth_input) # num_agents x h_dim
        
        # Local Attention
        
        for layer_index, layer in enumerate(self.Attn):
            temp = torch.clone(hidden_states_batch)
            hidden_states_batch = torch.cat(layer(hidden_states_batch, agents_per_sample),dim=0) 
            hidden_states_batch = hidden_states_batch + temp
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
   
        return hidden_states_batch

class MapSubNet(nn.Module):

    def __init__(self, h_dim):
        super(MapSubNet, self).__init__()
        
        self.h_dim = h_dim  
        self.num_attention_heads = NUM_ATTENTION_HEADS  
        self.depth = 2

        input_dim = 2 # Each point of the centerline has two attributes (xy)

        self.MLPs = nn.ModuleList([make_mlp([input_dim*CENTERLINE_LENGTH, self.h_dim // 4], batch_norm=True, dropout=DROPOUT), 
                                   make_mlp([self.h_dim // 4, self.h_dim], batch_norm=True, dropout=DROPOUT)])
        self.Attn = nn.ModuleList([MultiHeadSelfAttention(h_dim=self.h_dim // 4, num_heads=self.num_attention_heads),
                                   MultiHeadSelfAttention(h_dim=self.h_dim, num_heads=self.num_attention_heads)])
        self.Norms = nn.ModuleList([nn.LayerNorm(self.h_dim // 4), nn.LayerNorm(self.h_dim)])

        self.final_layer = nn.Linear(NUM_CENTERLINES*H_DIM,H_DIM)
        
    def forward(self, centerlines, centerlines_per_sample):
        batch_size = centerlines.shape[0] // NUM_CENTERLINES
        hidden_states_batch = centerlines.view(centerlines.shape[0],-1) # relevant_centerlines x (length · data_dim)
        
        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            temp = hidden_states_batch
            hidden_states_batch = torch.cat(layer(hidden_states_batch, centerlines_per_sample),dim=0) 
            hidden_states_batch = temp + hidden_states_batch
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)

        return hidden_states_batch
    
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

class InfoInteraction(nn.Module):
    def __init__(self, h_dim, dropout=0.1) -> None:
        super(InfoInteraction, self).__init__()

        self.h_dim = h_dim
        self.num_attention_heads = NUM_ATTENTION_HEADS
        
        self.cross_attn = nn.MultiheadAttention(self.h_dim, self.num_attention_heads, DROPOUT)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(self.h_dim)
        self.norm2 = nn.LayerNorm(self.h_dim)
        self.norm3 = nn.LayerNorm(self.h_dim)

        self.mlp = make_mlp([self.h_dim,self.h_dim],batch_norm=False)

    def forward(self, x_padding, y_padding):# , y_mask):

        cross_attn_output, _ = self.cross_attn(query=x_padding,
                                            key=y_padding,
                                            value=y_padding,
                                            attn_mask=None,
                                            key_padding_mask=None
                                            )

        x_padding = x_padding + self.dropout2(cross_attn_output)
        x_padding = self.norm2(x_padding)

        output = self.mlp(x_padding)

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
        
        self.num_modes = NUM_MODES
        norm = "BN"
        ng = 1

        pred = []
        for _ in range(self.num_modes):
            pred.append(
                nn.Sequential(
                    LinearRes(decoder_h_dim, decoder_h_dim, norm=norm, ng=ng),
                    nn.Linear(decoder_h_dim, PRED_LEN * DATA_DIM),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.confidences = nn.Sequential(LinearRes(decoder_h_dim, decoder_h_dim, norm=norm, ng=ng), 
                                         nn.Linear(decoder_h_dim, self.num_modes))
        
    def forward(self, latent_space):
        """_summary_

        Args:
            latent_space (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        batch_size = latent_space.shape[0]
        
        pred_traj_fake_rel = []
        
        for num_mode in range(self.num_modes):
            curr_pred_traj_fake_rel = self.pred[num_mode](latent_space)
            pred_traj_fake_rel.append(curr_pred_traj_fake_rel.contiguous().view(-1,PRED_LEN,DATA_DIM))
        
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim
        
        conf = self.confidences(latent_space) 
        conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()
            
        return pred_traj_fake_rel, conf
    
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

        self.spatial_embedding = nn.Linear(self.window_size*2, self.window_size*4)
        
        self.decoder = nn.LSTM(self.window_size*4, 
                               self.decoder_h_dim, 
                               num_layers=1)
  
        pred = []
        for _ in range(self.num_modes):
            pred.append(nn.Linear(self.decoder_h_dim,self.data_dim))
        self.hidden2pos = nn.ModuleList(pred) 

        norm = "BN"
        ng = 1
        
        self.confidences = nn.Sequential(LinearRes(decoder_h_dim, decoder_h_dim, norm=norm, ng=ng), 
                                         nn.Linear(decoder_h_dim, self.num_modes))
        
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
            state_tuple_c_ = torch.zeros(tuple(state_tuple_h_.shape)).cuda(state_tuple_h_.device)
            
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

        state_tuple_h = state_tuple_h.squeeze(0)
        conf = self.confidences(state_tuple_h)
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
        
        ## Physical 
        
        self.physical_encoder = MapSubNet(self.h_dim)
        mid_dim = (self.h_dim * self.num_centerlines + self.h_dim) // 2
        self.mlp_latentmap = make_mlp([self.h_dim * self.num_centerlines, mid_dim, self.h_dim], batch_norm=True, dropout=DROPOUT)
        
        ## Information interaction
        
        self.info_interaction = InfoInteraction(self.h_dim)
        
        ## Global interaction
        
        # ln = True
        # num_inds = 8
        
        # self.global_interaction = ISAB(self.h_dim, self.h_dim, self.num_attention_heads, num_inds, ln=ln)
        # self.global_interaction = PMA(self.h_dim, self.num_attention_heads, self.num_modes, ln=ln)

        # Decoder
        
        self.decoder = Temporal_Multimodal_Decoder(self.h_dim*2)
        # self.decoder = Multimodal_Decoder(self.h_dim)
            
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

        batch_size = seq_start_end.shape[0]
        
        # Encoder
        
        ## Actor encoder
        
        centers = obs_traj[-1,:,:] # x,y (abs coordinates)
        agents_per_sample = (seq_start_end[:,1] - seq_start_end[:,0]).cpu().detach().numpy()
        
        agents_features = self.social_encoder(obs_traj_rel, agents_per_sample)
        out_agent_gnn = self.agent_gnn(agents_features, centers, agents_per_sample)

        if np.any(agent_idx): # Single agent
            social_info = out_agent_gnn[agent_idx,:]
        
        ## Map encoder

        _, num_centerlines, points_centerline, data_dim = relevant_centerlines.shape
        relevant_centerlines = relevant_centerlines.view(-1, points_centerline, data_dim)
   
        rows_mask, _ = torch.where(relevant_centerlines[:,:,0] == 0.0)
        mask = torch.unique(rows_mask) # Rows where we have padded-centerlines
        relevant_centerlines_mask = torch.zeros([relevant_centerlines.shape[0]], dtype=torch.bool)
        relevant_centerlines_mask[mask] = True
        relevant_centerlines_mask_inverted = ~relevant_centerlines_mask # Non-padded centerlines (so, relevant) to True
        
        centerlines_per_sample = [] # Relevant centerlines (non-padded) per sequence
        num_current_centerlines = 0

        for i in range(relevant_centerlines_mask.shape[0]+1):
            if i % NUM_CENTERLINES == 0 and i > 0: # Next traffic scenario
                centerlines_per_sample.append(num_current_centerlines)
                num_current_centerlines = 0
                
                if i == relevant_centerlines_mask.shape[0]:
                    break
            if relevant_centerlines_mask_inverted[i]: # Non-masked
                num_current_centerlines += 1
                
        centerlines_per_sample = np.array(centerlines_per_sample)
        relevant_centerlines = relevant_centerlines[relevant_centerlines_mask_inverted,:,:]

        aux_physical_info = self.physical_encoder(relevant_centerlines, centerlines_per_sample)
        physical_info = torch.zeros((batch_size*num_centerlines, self.h_dim), device=aux_physical_info.device)
        physical_info[relevant_centerlines_mask_inverted] = aux_physical_info
        physical_info = physical_info.contiguous().view(batch_size,self.h_dim*self.num_centerlines)
        physical_info = self.mlp_latentmap(physical_info)
        
        ## Information interaction
        
        # crossed_info = self.info_interaction(social_info, physical_info)
        crossed_info = torch.cat([social_info, 
                                               physical_info], 
                                               dim=1)

        # Decoder
        
        decoder_h = crossed_info.unsqueeze(0)
        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(obs_traj.device)

        state_tuple = (decoder_h, decoder_c)
        
        traj_agent_abs = obs_traj[-WINDOW_SIZE:, agent_idx, :]
        traj_agent_abs_rel = obs_traj_rel[-WINDOW_SIZE:, agent_idx, :]

        pred_traj_fake_rel, conf = self.decoder(traj_agent_abs, traj_agent_abs_rel, state_tuple)
        # pred_traj_fake_rel, conf = self.decoder(crossed_info)
        
        return pred_traj_fake_rel, conf