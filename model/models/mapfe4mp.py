#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## MAPFE4MP: Map Features For Efficient Motion Prediction 

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import math
import numpy as np
import pdb

# DL & Math imports

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse

# Custom imports

from model.modules.layers import Linear, LinearRes

#######################################

# Global variables

DATA_DIM = 2
NUM_MODES = 6

OBS_LEN = 20
PRED_LEN = 30
NUM_PLAUSIBLE_AREA_POINTS = 512
CENTERLINE_LENGTH = 40

NUM_ATTENTION_HEADS = 4
H_DIM_PHYSICAL = 128
H_DIM_SOCIAL = 128
EMBEDDING_DIM = 16
CONV_FILTERS = 60 # 60

APPLY_DROPOUT = True
DROPOUT = 0.4

HEAD = "SingleLinear" # SingleLinear, MultiLinear, Non-Autoregressive

def make_mlp(dim_list, activation_function="ReLU", batch_norm=False, dropout=0.0):
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

        if activation_function == "ReLU":
            layers.append(nn.ReLU())
        elif activation_function == "GELU":
            layers.append(nn.GELU())
        elif activation_function == "Tanh":
            layers.append(nn.Tanh())

        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))

        index += 1
    return nn.Sequential(*layers)

class MotionEncoder(nn.Module):
    def __init__(self, h_dim):
        super(MotionEncoder, self).__init__()

        self.input_size = DATA_DIM
        self.hidden_size = h_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

    def forward(self, lstm_in):
        """_summary_

        Args:
            lstm_in (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        lstm_hidden_state = torch.randn(
            self.num_layers, lstm_in.shape[1], self.hidden_size, device=lstm_in.device)
        lstm_cell_state = torch.randn(
            self.num_layers, lstm_in.shape[1], self.hidden_size, device=lstm_in.device)
        lstm_hidden = (lstm_hidden_state, lstm_cell_state)

        lstm_out, (lstm_hidden, lstm_cell) = self.lstm(lstm_in, lstm_hidden)
        
        # lstm_out is the hidden state over all time steps from the last LSTM layer
        # In this case, only the features of the last time step are used

        if APPLY_DROPOUT: lstm_hidden = F.dropout(lstm_hidden, p=DROPOUT, training=self.training)
        lstm_hidden_ = lstm_hidden.squeeze(dim=0)
        return lstm_hidden_

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

        if APPLY_DROPOUT: self.multihead_attention = nn.MultiheadAttention(self.latent_size, 4, dropout=DROPOUT)
        else: self.multihead_attention = nn.MultiheadAttention(self.latent_size, 4)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa

        test = True
        if len(agents_per_sample) > 1 or test:
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
        else:
            pdb.set_trace()
            agents_per_sample_ = tuple(np.ones(agents_per_sample,dtype=int))
            att_in = torch.split(att_in, agents_per_sample_)
            for i, sample in enumerate(att_in):
                # Add the batch dimension (this has to be the second dimension, because attention requires it)
                att_in_formatted = sample.unsqueeze(1)
                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted)

                # Remove the "1" batch dimension
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)

        return att_out_batch

class Centerline_Encoder(nn.Module):
    def __init__(self, h_dim, kernel_size=3):
        super(Centerline_Encoder, self).__init__()

        self.data_dim = DATA_DIM
        self.num_filters = CONV_FILTERS
        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.lane_length = CENTERLINE_LENGTH

        self.pooling_size = 2
        self.tanh = torch.nn.Tanh()
        
        self.bn0 = nn.BatchNorm1d(self.data_dim)
        self.conv1 = nn.Conv1d(self.data_dim,self.num_filters,self.kernel_size)
        self.maxpooling1 = nn.MaxPool1d(self.pooling_size)
        self.bn1 = nn.BatchNorm1d(self.num_filters)
        
        self.conv2 = nn.Conv1d(self.num_filters,self.num_filters,self.kernel_size)
        self.maxpooling2 = nn.MaxPool1d(self.pooling_size)
        self.bn2 = nn.BatchNorm1d(self.num_filters)

        # Compute centerline length after convolution+pooling

        num_convs = num_poolings = 0
        for key in self._modules.keys():
            if "conv" in key:
                num_convs += 1
            elif "pooling" in key:
                num_poolings += 1

        sub_ = 0
        for i in range(1,num_convs+1):
            sub_ += pow(2,i)

        aux_length = int(math.floor((self.lane_length-sub_)/pow(2,num_poolings)))
        assert aux_length >= 2 # TODO: Minimum number here?

        # N convolutional filters * aux_length

        self.flatten = torch.nn.Flatten(start_dim=1,end_dim=-1)
        self.linear = nn.Linear(self.num_filters*aux_length,self.h_dim)
        
        # TODO: What is better, centerline encoder using conv+pooling, or MLP?
        # mid_dim = math.ceil((self.lane_length*self.data_dim + self.h_dim)/2)
        # dims = [self.lane_length*self.data_dim, mid_dim, self.h_dim]
        # self.mlp_centerlines = make_mlp(dims,
        #                      activation_function="Tanh",
        #                      batch_norm=True,
        #                      dropout=DROPOUT)

    def forward(self, phy_info):
        """_summary_

        Args:
            phy_info (_type_): _description_

        Returns:
            _type_: _description_
        """

        num_centerlines = phy_info.shape[0]
        phy_info_ = torch.clone(phy_info.permute(0,2,1))
        phy_info_ = self.bn0(phy_info_)
        
        phy_info_ = self.conv1(phy_info.permute(0,2,1))
        phy_info_ = self.bn1(phy_info_)
        phy_info_ = self.tanh(phy_info_)
        phy_info_ = self.maxpooling1(phy_info_)
         
        phy_info_ = self.conv2(phy_info_)
        phy_info_ = self.bn2(phy_info_)
        phy_info_ = self.tanh(phy_info_)
        phy_info_ = self.maxpooling2(phy_info_)

        phy_info_ = self.linear(self.flatten(phy_info_))
        phy_info_ = F.dropout(phy_info_, p=DROPOUT, training=self.training)

        # phy_info_ = phy_info_.permute(0,2,1)
        # phy_info_ = phy_info_.contiguous().view(num_centerlines, -1)
        # phy_info_ = self.mlp_centerlines(phy_info_)

        if torch.any(phy_info_.isnan()):
            pdb.set_trace()
            
        return phy_info_
        
class Multimodal_Decoder(nn.Module):
    def __init__(self, decoder_h_dim):
        super(Multimodal_Decoder, self).__init__()

        self.data_dim = DATA_DIM
        self.pred_len = PRED_LEN

        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.num_modes = NUM_MODES

        self.traj_points = self.num_modes*2 # Vector length per step (6 modes x 2 (x,y))
        self.spatial_embedding = nn.Linear(self.traj_points, self.embedding_dim)
        
        self.decoder = nn.LSTM(self.embedding_dim, self.decoder_h_dim, 1)
  
        if HEAD == "MultiLinear":
            pred = []
            for _ in range(self.num_modes):
                # pred.append(nn.Linear(self.decoder_h_dim, self.data_dim))
                pred.append(Linear(self.decoder_h_dim, self.data_dim))
            self.hidden2pos = nn.ModuleList(pred) 
        elif HEAD == "SingleLinear":
            self.hidden2pos = nn.Linear(self.decoder_h_dim, self.traj_points)
            # self.hidden2pos = Linear(self.decoder_h_dim, self.traj_points)
        elif HEAD == "Non-Autoregressive":
            pred = []
            for _ in range(self.num_modes):
                # pred.append(nn.Linear(self.decoder_h_dim, self.data_dim))
                pred.append(Linear(self.decoder_h_dim, self.data_dim*self.pred_len))
            self.hidden2pos = nn.ModuleList(pred) 

        self.confidences = nn.Linear(self.decoder_h_dim, self.num_modes)

    def forward(self, last_obs, last_obs_rel, state_tuple):
        """_summary_

        Args:
            last_obs (_type_): _description_
            last_obs_rel (_type_): _description_
            state_tuple (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # TODO: Check the view method, not advisable!

        batch_size, data_dim = last_obs.shape
        last_obs_rel = last_obs_rel.view(1,batch_size,1,data_dim)
        last_obs_rel = last_obs_rel.repeat_interleave(self.num_modes, dim=2)
        last_obs_rel = last_obs_rel.view(1, batch_size, -1)

        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(last_obs_rel.contiguous().view(batch_size, -1))) 
        if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)

        state_tuple_h, state_tuple_c = state_tuple
        state_tuple_h_ = torch.clone(state_tuple_h)
        state_tuple_c_ = torch.clone(state_tuple_c)

        if HEAD == "Non-Autoregressive":
            output, (state_tuple_h, state_tuple_c) = self.decoder(decoder_input, (state_tuple_h, state_tuple_c)) 

            pred_traj_fake_rel = []
            for num_mode in range(self.num_modes):
                rel_pos_ = self.hidden2pos[num_mode](state_tuple_h.contiguous().view(-1, self.decoder_h_dim))
                pred_traj_fake_rel.append(rel_pos_.contiguous().view(batch_size,self.pred_len,self.data_dim).unsqueeze(0))

            pred_traj_fake_rel = torch.stack(pred_traj_fake_rel,dim=0)
            pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
            pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim

        else:
            for _ in range(self.pred_len):
                output, (state_tuple_h_, state_tuple_c_) = self.decoder(decoder_input, (state_tuple_h_, state_tuple_c_)) 

                if HEAD == "MultiLinear":
                    rel_pos = []
                    for num_mode in range(self.num_modes):
                        rel_pos_ = self.hidden2pos[num_mode](state_tuple_h_.contiguous().view(-1, self.decoder_h_dim)) # 2
                        rel_pos.append(rel_pos_)

                    rel_pos = torch.cat(rel_pos,dim=1)   
                elif HEAD == "SingleLinear":
                    rel_pos = self.hidden2pos(state_tuple_h_.contiguous().view(-1, self.decoder_h_dim))

                decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
                if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
                decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)           
                pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

            pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
            pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
            pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim

        conf = self.confidences(state_tuple_h.contiguous().view(-1, self.decoder_h_dim))

        conf = torch.softmax(conf, dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()

        return pred_traj_fake_rel, conf

class TrajectoryGenerator(nn.Module):
    def __init__(self, PHYSICAL_CONTEXT="dummy"):
        super(TrajectoryGenerator, self).__init__()

        self.physical_context = PHYSICAL_CONTEXT

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.h_dim_social = H_DIM_SOCIAL
        self.h_dim_physical = H_DIM_PHYSICAL
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.data_dim = DATA_DIM
        self.num_modes = NUM_MODES
        self.num_plausible_area_points = NUM_PLAUSIBLE_AREA_POINTS

        # Encoder

        ## Social 

        self.social_encoder = MotionEncoder(h_dim=self.h_dim_social)
        self.agent_gnn = GNN(h_dim=self.h_dim_social)
        self.sattn = MultiheadSelfAttention(h_dim=self.h_dim_social,
                                            num_heads=self.num_attention_heads)

        ## Physical 

        # mid_dim = math.ceil((self.num_plausible_area_points*self.data_dim + self.h_dim)/2)
        # plausible_area_encoder_dims = [self.num_plausible_area_points*self.data_dim, mid_dim, self.h_dim]
        # self.bn = nn.BatchNorm1d(self.num_plausible_area_points*self.data_dim)
        # self.plausible_area_encoder = make_mlp(plausible_area_encoder_dims,
        #                                        activation_function="Tanh")
        self.centerline_encoder = Centerline_Encoder(h_dim=self.h_dim_social)
        self.centerline_gnn = GNN(h_dim=self.h_dim_physical)
        self.pattn = MultiheadSelfAttention(h_dim=self.h_dim_physical,
                                            num_heads=self.num_attention_heads)
   
        # Decoder
        
        # TODO: Required MLP to merge the concatenated information (Social + Physical) before the decoder?

        # mlp_decoder_context_input_dim = self.h_dim # After GNN and MHSA
 
        # mlp_decoder_context_dims = [mlp_decoder_context_input_dim, self.mlp_dim, self.h_dim]
        # self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims,
        #                                     activation_function="Tanh")
        
        if PHYSICAL_CONTEXT == "social":
            # encoded target pos and vel + social info
            self.concat_h_dim = 3 * self.h_dim_social 
        elif PHYSICAL_CONTEXT == "oracle":
            # encoded target pos and vel + social info + most plausible centerline encoded
            # self.concat_h_dim = 3 * self.h_dim_social + self.h_dim_physical
            
            self.concat_h_dim = self.h_dim_social + self.h_dim_physical
        elif PHYSICAL_CONTEXT == "plausible_centerlines+area":
            # encoded target pos and vel + social info + map information encoded
            self.concat_h_dim = 3 * self.h_dim_social + 6 * self.h_dim_physical
 
        self.decoder = Multimodal_Decoder(decoder_h_dim=self.concat_h_dim)  

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
        
        # Data aug (TODO: Do this in the seq_collate function)
        
        obs_traj = self.add_noise(obs_traj, factor=0.5)
        obs_traj_rel = self.add_noise(obs_traj_rel, factor=0.01)
        if self.physical_context == "oracle":
            phy_info = self.add_noise(phy_info)
        elif self.physical_context == "plausible_centerlines+area":
            # phy_info = self.add_noise(phy_info)
            relevant_centerlines = self.add_noise(relevant_centerlines)
        
        # Encoder

        encoded_obs_traj = self.social_encoder(obs_traj)
        encoded_obs_traj_rel = self.social_encoder(obs_traj_rel)

        target_agent_encoded_obs_traj = encoded_obs_traj[agent_idx,:]
        target_agent_encoded_obs_traj_rel = encoded_obs_traj_rel[agent_idx,:]

        ## Social information

        centers = obs_traj[-1,:,:] # x,y (abs coordinates)
        agents_per_sample = (seq_start_end[:,1] - seq_start_end[:,0]).cpu().detach().numpy()

        out_agent_gnn = self.agent_gnn(encoded_obs_traj_rel, centers, agents_per_sample)
        out_self_attention = self.sattn(out_agent_gnn, agents_per_sample)
        encoded_social_info = torch.cat(out_self_attention,dim=0) # batch_size · num_agents x hidden_dim_social

        if np.any(agent_idx): # single agent
            encoded_social_info = encoded_social_info[agent_idx,:]
 
        ## Physical information

        last_pos = obs_traj[-1, agent_idx, :]
        last_pos_rel = obs_traj_rel[-1, agent_idx, :]

        if self.physical_context == "social":
            mlp_decoder_context_input = torch.cat([target_agent_encoded_obs_traj.contiguous().view(-1,self.h_dim_social), 
                                                   target_agent_encoded_obs_traj_rel.contiguous().view(-1,self.h_dim_social), 
                                                   encoded_social_info.contiguous().view(-1,self.h_dim_social)], 
                                                   dim=1)

            decoder_h = mlp_decoder_context_input.unsqueeze(0)
            decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)
            state_tuple = (decoder_h, decoder_c)

        elif self.physical_context == "oracle":
            encoded_phy_info = self.centerline_encoder(phy_info)
            # mlp_decoder_context_input = torch.cat([target_agent_encoded_obs_traj.contiguous().view(-1,self.h_dim_social), 
            #                                        target_agent_encoded_obs_traj_rel.contiguous().view(-1,self.h_dim_social), 
            #                                        encoded_social_info.contiguous().view(-1,self.h_dim_social), 
            #                                        encoded_phy_info.contiguous().view(-1,self.h_dim_physical)], 
            #                                        dim=1)

            mlp_decoder_context_input = torch.cat([encoded_social_info, 
                                                   encoded_phy_info], 
                                                   dim=1)
            
            decoder_h = mlp_decoder_context_input.unsqueeze(0)
            decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)
            state_tuple = (decoder_h, decoder_c)

        elif self.physical_context == "plausible_centerlines+area":
            NUM_CENTERLINES = 6 # TODO: This should be variable
            centerlines_per_sample = NUM_CENTERLINES * np.ones(batch_size,dtype=int)

            relevant_centerlines_ = relevant_centerlines.permute(1,0,2,3)
            relevant_centerlines_concat = relevant_centerlines_.contiguous().view(-1,CENTERLINE_LENGTH,self.data_dim)
            
            encoded_centerlines = self.centerline_encoder(relevant_centerlines_concat)
            out_centerlines_gnn = self.centerline_gnn(encoded_centerlines, centers, centerlines_per_sample)
            encoded_phy_info = self.pattn(out_centerlines_gnn, centerlines_per_sample)
            encoded_phy_info = torch.stack(encoded_phy_info,dim=0).view(batch_size,-1)

            mlp_decoder_context_input = torch.cat([target_agent_encoded_obs_traj, 
                                                   target_agent_encoded_obs_traj_rel, 
                                                   encoded_social_info, 
                                                   encoded_phy_info], 
                                                   dim=1)
 
            decoder_h = mlp_decoder_context_input.unsqueeze(0)
            decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)

            state_tuple = (decoder_h, decoder_c)

        pred_traj_fake_rel, conf = self.decoder(last_pos, last_pos_rel, state_tuple)
        
        if torch.any(pred_traj_fake_rel.isnan()) or torch.any(conf.isnan()):
            pdb.set_trace()

        return pred_traj_fake_rel, conf