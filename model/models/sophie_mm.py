import os
import math
import random
import copy
import time
import numpy as np
import pdb
from model.datasets.argoverse.dataset import PHYSICAL_CONTEXT

from model.modules.attention import MultiHeadAttention
from model.modules.layers import Linear

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse

DATA_DIM = 2
NUM_MODES = 6

OBS_LEN = 20
PRED_LEN = 30
NUM_PLAUSIBLE_AREA_POINTS = 512

MLP_DIM = 64
H_DIM = 128
EMBEDDING_DIM = 16
CONV_FILTERS = 60 # 60

APPLY_DROPOUT = True
DROPOUT = 0.4

HEAD = "SingleLinear" # SingleLinear, MultiLinear, Non-Autoregressive

def make_mlp(dim_list,activation_function="ReLU"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if activation_function == "ReLU":
            layers.append(nn.ReLU())
        elif activation_function == "GELU":
            layers.append(nn.GELU())
        elif activation_function == "Tanh":
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class EncoderLSTM_CRAT(nn.Module):
    def __init__(self):
        super(EncoderLSTM_CRAT, self).__init__()

        self.input_size = DATA_DIM
        self.hidden_size = H_DIM
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

    def forward(self, lstm_in):
        # lstm_in are all agents over all samples in the current batch
        # Format for LSTM has to be has to be (batch_size, timeseries_length, latent_size), because batch_first=True

        # Initialize the hidden state.
        # lstm_in.shape[0] corresponds to the number of all agents in the current batch
        lstm_hidden_state = torch.randn(
            self.num_layers, lstm_in.shape[1], self.hidden_size, device=lstm_in.device)
        lstm_cell_state = torch.randn(
            self.num_layers, lstm_in.shape[1], self.hidden_size, device=lstm_in.device)
        lstm_hidden = (lstm_hidden_state, lstm_cell_state)

        lstm_out, (lstm_hidden, lstm_cell) = self.lstm(lstm_in, lstm_hidden)
        
        # lstm_out is the hidden state over all time steps from the last LSTM layer
        # In this case, only the features of the last time step are used

        if APPLY_DROPOUT: lstm_hidden = F.dropout(lstm_hidden, p=DROPOUT, training=self.training)

        # return lstm_out[-1, :, :]
        return lstm_hidden.view(-1,self.hidden_size)

class AgentGnn_CRAT(nn.Module):
    def __init__(self):
        super(AgentGnn_CRAT, self).__init__()

        self.latent_size = H_DIM

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
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        return gnn_out

class MultiheadSelfAttention_CRAT(nn.Module):
    def __init__(self):
        super(MultiheadSelfAttention_CRAT, self).__init__()

        self.latent_size = H_DIM

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
        
class Multimodal_DecoderLSTM(nn.Module):
    def __init__(self, PHYSICAL_CONTEXT="dummy"):
        super().__init__()

        self.data_dim = DATA_DIM
        self.pred_len = PRED_LEN

        if PHYSICAL_CONTEXT != "dummy":
            self.h_dim = H_DIM*2
        else:
            self.h_dim = H_DIM

        self.mlp_dim = MLP_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.num_modes = NUM_MODES

        self.traj_points = self.num_modes*2 # Vector length per step (6 modes x 2 (x,y))
        self.spatial_embedding = nn.Linear(self.traj_points, self.embedding_dim)
        
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
  
        if HEAD == "MultiLinear":
            pred = []
            for _ in range(self.num_modes):
                # pred.append(nn.Linear(self.h_dim, self.data_dim))
                pred.append(Linear(self.h_dim, self.data_dim))
            self.hidden2pos = nn.ModuleList(pred) 
        elif HEAD == "SingleLinear":
            self.hidden2pos = nn.Linear(self.h_dim, self.traj_points)
            # self.hidden2pos = Linear(self.h_dim, self.traj_points)
        elif HEAD == "Non-Autoregressive":
            pred = []
            for _ in range(self.num_modes):
                # pred.append(nn.Linear(self.h_dim, self.data_dim))
                pred.append(Linear(self.h_dim, self.data_dim*self.pred_len))
            self.hidden2pos = nn.ModuleList(pred) 

        self.confidences = nn.Linear(self.h_dim, self.num_modes)

    def forward(self, last_obs, last_obs_rel, state_tuple):
        """
        last_obs (1, b, 2)
        last_obs_rel (1, b, 2)
        state_tuple: h and c
            h : c : (1, b, self.h_dim)
        """

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
                rel_pos_ = self.hidden2pos[num_mode](state_tuple_h.contiguous().view(-1, self.h_dim))
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
                        rel_pos_ = self.hidden2pos[num_mode](state_tuple_h_.contiguous().view(-1, self.h_dim))
                        rel_pos.append(rel_pos_)

                    rel_pos = torch.cat(rel_pos,dim=1)   
                elif HEAD == "SingleLinear":
                    rel_pos = self.hidden2pos(state_tuple_h_.contiguous().view(-1, self.h_dim))

                decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
                if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
                decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)           
                pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

            pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
            pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, self.num_modes, -1)
            pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) # batch_size, num_modes, pred_len, data_dim

        conf = self.confidences(state_tuple_h.contiguous().view(-1, self.h_dim))

        conf = torch.softmax(conf, dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()

        return pred_traj_fake_rel, conf

class Unimodal_DecoderLSTM(nn.Module):
    def __init__(self, PHYSICAL_CONTEXT="dummy"):
        super().__init__()

        self.data_dim = DATA_DIM
        self.pred_len = PRED_LEN

        if PHYSICAL_CONTEXT == "dummy":
            self.h_dim = H_DIM
        elif PHYSICAL_CONTEXT == "plausible_centerlines+area":
            # self.h_dim = H_DIM*2
            self.h_dim = H_DIM*3
            # self.h_dim = H_DIM
        else:
            self.h_dim = H_DIM*2

        self.mlp_dim = MLP_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.num_modes = NUM_MODES

        self.spatial_embedding = nn.Linear(self.data_dim, self.embedding_dim)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)

        self.hidden2pos = nn.Linear(self.h_dim, self.data_dim) # TODO: More complex?
    
    def forward(self, last_obs, last_obs_rel, state_tuple):
        """
        last_obs (1, b, 2)
        last_obs_rel (1, b, 2)
        state_tuple: h and c
            h : c : (1, b, self.h_dim)
        """
        
        state_tuple_h, state_tuple_c = state_tuple
        state_tuple_h_ = torch.clone(state_tuple_h) # bs x 384
        state_tuple_c_ = torch.clone(state_tuple_c) # bs x 384 (randn)

        batch_size, data_dim = last_obs.shape
        last_obs_rel = last_obs_rel.view(1,batch_size,1,data_dim)

        pred_traj_fake_rel = []
        decoder_input = F.tanh(self.spatial_embedding(last_obs_rel.contiguous().view(batch_size, -1))) 
        if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)

        for _ in range(self.pred_len):
            output, (state_tuple_h_, state_tuple_c_) = self.decoder(decoder_input, (state_tuple_h_, state_tuple_c_)) 

            rel_pos = self.hidden2pos(state_tuple_h_.contiguous().view(-1, self.h_dim))

            decoder_input = F.tanh(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            if APPLY_DROPOUT: decoder_input = F.dropout(decoder_input, p=DROPOUT, training=self.training)
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)           
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len, batch_size, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2) # batch_size, pred_len, data_dim

        return pred_traj_fake_rel

class TemporalDecoderLSTM(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

        self.data_dim = DATA_DIM # x,y
        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.hidden2pos = nn.Linear(self.h_dim, self.data_dim)

        self.spatial_embedding = nn.Linear(self.obs_len*2, self.embedding_dim)
        self.ln1 = nn.LayerNorm(self.obs_len*2)
        self.ln2 = nn.LayerNorm(self.h_dim)

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim)

    def forward(self, traj_abs, traj_rel, state_tuple):
        """ 
        """

        num_agents = traj_abs.size(1)

        state_tuple_h, state_tuple_c = state_tuple
        state_tuple_h_ = torch.clone(state_tuple_h)
        state_tuple_c_ = torch.clone(state_tuple_c)
 
        pred_traj_fake_rel = []
        decoder_input = F.tanh(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(num_agents, -1))))
        decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)

        for _ in range(self.pred_len):
            output, (state_tuple_h_, state_tuple_c_) = self.decoder(decoder_input, (state_tuple_h_, state_tuple_c_))

            rel_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim)))
            traj_rel = torch.roll(traj_rel, -1, dims=(0))
            traj_rel[-1] = rel_pos
            pred_traj_fake_rel.append(rel_pos.contiguous().view(num_agents,-1))

            decoder_input = F.tanh(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(num_agents, -1))))
            decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2)

        return pred_traj_fake_rel

class Centerline_Encoder(nn.Module):
    def __init__(self):
        super(Centerline_Encoder, self).__init__()

        self.data_dim = DATA_DIM
        self.num_filters = CONV_FILTERS
        self.kernel_size = 3
        self.lane_length = 40

        self.pooling_size = 2

        self.h_dim = H_DIM

        self.conv1 = nn.Conv1d(self.data_dim,self.num_filters,self.kernel_size)
        self.maxpooling1 = nn.MaxPool1d(self.pooling_size)

        self.conv2 = nn.Conv1d(self.num_filters,self.num_filters,self.kernel_size)
        self.maxpooling2 = nn.MaxPool1d(self.pooling_size)

        self.conv3 = nn.Conv1d(self.num_filters,self.num_filters,self.kernel_size)

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

        self.linear = nn.Linear(self.num_filters*aux_length,self.h_dim)
        self.bn = nn.BatchNorm1d(self.h_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, phy_info):
        """
        """

        batch_size = phy_info.shape[0]

        phy_info_ = self.conv1(phy_info.permute(0,2,1))
        phy_info_ = self.maxpooling1(phy_info_)
        phy_info_ = self.conv2(phy_info_)
        phy_info_ = self.maxpooling2(phy_info_)
        phy_info_ = self.conv3(phy_info_)

        phy_info_ = self.linear(phy_info_.view(batch_size,-1))
        phy_info_ = self.bn(phy_info_)
        phy_info_ = self.tanh(phy_info_)
        phy_info_ = F.dropout(phy_info_, p=DROPOUT, training=self.training)

        return phy_info_
        
class TrajectoryGenerator(nn.Module):
    def __init__(self, PHYSICAL_CONTEXT="dummy"):
        super(TrajectoryGenerator, self).__init__()

        self.physical_context = PHYSICAL_CONTEXT

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.data_dim = DATA_DIM
        self.num_modes = NUM_MODES
        self.num_plausible_area_points = NUM_PLAUSIBLE_AREA_POINTS

        # Encoder

        ## Social 

        self.social_encoder = EncoderLSTM_CRAT()
        self.agent_gnn = AgentGnn_CRAT()
        self.sattn = MultiheadSelfAttention_CRAT()

        ## Physical 

        mid_dim = math.ceil((self.num_plausible_area_points*self.data_dim + self.h_dim)/2)
        plausible_area_encoder_dims = [self.num_plausible_area_points*self.data_dim, mid_dim, self.h_dim]
        self.bn = nn.BatchNorm1d(self.num_plausible_area_points*self.data_dim)
        self.plausible_area_encoder = make_mlp(plausible_area_encoder_dims,
                                               activation_function="Tanh")
        self.centerline_encoder = Centerline_Encoder()
   
        # Decoder

        mlp_decoder_context_input_dim = self.h_dim # After GNN and MHSA
 
        mlp_decoder_context_dims = [mlp_decoder_context_input_dim, self.mlp_dim, self.h_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims,
                                            activation_function="Tanh")

        if self.physical_context == "plausible_centerlines+area":
            self.decoder = Unimodal_DecoderLSTM(PHYSICAL_CONTEXT=self.physical_context)
            self.confidences = nn.Sequential(nn.Linear(self.pred_len*self.data_dim,32),
                                             nn.ReLU(),
                                             nn.Linear(32,1))     
        else:
            self.decoder = Multimodal_DecoderLSTM(PHYSICAL_CONTEXT=self.physical_context)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, agent_idx, phy_info=None, relevant_centerlines=[]):
        """
        """

        batch_size = seq_start_end.shape[0]

        # Encoder

        final_encoder_h = self.social_encoder(obs_traj_rel)

        ## Social information (GNN + MHSA)

        centers = obs_traj[-1,:,:] # x,y (abs coordinates)
        agents_per_sample = (seq_start_end[:,1] - seq_start_end[:,0]).cpu().detach().numpy()

        out_agent_gnn = self.agent_gnn(final_encoder_h, centers, agents_per_sample)
        out_self_attention = self.sattn(out_agent_gnn, agents_per_sample)

        encoded_social_info = torch.cat(out_self_attention,dim=0) # batch_size · num_agents x num_inputs_decoder · hidden_dim

        if np.any(agent_idx): # single agent
            encoded_social_info = encoded_social_info[agent_idx,:]
 
        # Decoder

        ## Get agent last observations (both abs (around 0,0) and rel-rel)

        last_pos = obs_traj[-1, agent_idx, :]
        last_pos_rel = obs_traj_rel[-1, agent_idx, :]

        traj_agent_abs = obs_traj[:, agent_idx, :]
        traj_agent_abs_rel = obs_traj_rel[:, agent_idx, :]

        if self.physical_context == "dummy":
            mlp_decoder_context_input = encoded_social_info

            decoder_h = mlp_decoder_context_input.unsqueeze(0)
            decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)
            state_tuple = (decoder_h, decoder_c)

            ## Predict trajectories

            pred_traj_fake_rel, conf = self.decoder(last_pos, last_pos_rel, state_tuple)

        ## Include map information
        
        elif self.physical_context == "oracle":
            encoded_phy_info = self.centerline_encoder(phy_info)
            mlp_decoder_context_input = torch.cat([encoded_social_info.contiguous().view(-1,self.h_dim),
                                                   encoded_phy_info.contiguous().view(-1,self.h_dim)],
                                                   dim=1)

            decoder_h = mlp_decoder_context_input.unsqueeze(0)
            decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)
            state_tuple = (decoder_h, decoder_c)

            ## Predict trajectories

            pred_traj_fake_rel, conf = self.decoder(last_pos, last_pos_rel, state_tuple)

        elif self.physical_context == "plausible_centerlines+area":
            # Static map info

            plausible_area = torch.clone(phy_info)
            static_map_info = self.bn(plausible_area.view(batch_size,-1))
            encoded_static_map_info = self.plausible_area_encoder(static_map_info)

            pred_traj_fake_rel = []
        
            for mode in range(NUM_MODES):

                # Dynamic map info (one centerline per iteration -> Max Num_Modes iterations)

                current_centerlines = relevant_centerlines[mode,:,:,:]

                encoded_centerlines_info = self.centerline_encoder(current_centerlines)

                # Concatenate social info, static map info and current centerline info

                mlp_decoder_context_input = torch.cat([encoded_social_info.contiguous().view(-1,self.h_dim), 
                                                       encoded_static_map_info.contiguous().view(-1,self.h_dim),
                                                       encoded_centerlines_info.contiguous().view(-1,self.h_dim)], 
                                                       dim=1)

                decoder_h = mlp_decoder_context_input.unsqueeze(0)
                decoder_c = torch.randn(tuple(decoder_h.shape)).cuda(obs_traj.device)

                state_tuple = (decoder_h, decoder_c)

                ## Predict trajectories

                curr_pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
                curr_pred_traj_fake_rel = curr_pred_traj_fake_rel.unsqueeze(0)
                pred_traj_fake_rel.append(curr_pred_traj_fake_rel)

            pred_traj_fake_rel = torch.cat(pred_traj_fake_rel,dim=0)
            pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim

            # Compute confidences

            conf = self.confidences(pred_traj_fake_rel.contiguous().view(batch_size,self.num_modes,-1))
            conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
            if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
                pdb.set_trace()

        return pred_traj_fake_rel, conf