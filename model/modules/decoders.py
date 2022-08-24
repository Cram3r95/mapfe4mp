#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Decoder functions

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo, Miguel Eduardo Ortiz Huamaní and Marcos V. Conde
"""

# General purpose imports

import pdb

# DL & Math imports

import torch
from torch import nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    """
    """
    def __init__(self, input_len=20, output_len=30, h_dim=64, embedding_dim=16, 
                 num_layers=1, bidirectional=False, dropout=0.5, current_cuda="cuda:0",
                 use_rel_disp=False):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.ln1 = nn.LayerNorm(2)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
        self.ln2 = nn.LayerNorm(self.h_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(last_pos_rel))) # 16
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) #
            rel_pos = self.output_activation(self.hidden2pos(self.ln2(output.view(-1, self.h_dim))))# + last_pos_rel # 32 -> 2
            curr_pos = rel_pos + last_pos
            embedding_input = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(embedding_input)))
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(npeds,-1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class TemporalDecoderLSTM(nn.Module):
    """
    """
    def __init__(self, input_len=20, output_len=30, h_dim=64, embedding_dim=16, 
                 num_layers=1, bidirectional=False, dropout=0.5, current_cuda="cuda:0",
                 use_rel_disp=False):
        super().__init__()

        self.data_dim = 2 # x,y
        self.pred_len = output_len
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.current_cuda = current_cuda
        self.use_rel_disp = use_rel_disp

        if bidirectional: self.D = 2
        else: self.D = 1

        self.hidden2pos = nn.Linear(self.h_dim, self.data_dim)

        self.spatial_embedding = nn.Linear(input_len*2, self.embedding_dim)
        self.ln1 = nn.LayerNorm(input_len*2)
        self.ln2 = nn.LayerNorm(self.h_dim)

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, num_layers=num_layers,
                               dropout=dropout, bidirectional=bidirectional)

    def forward(self, traj_abs, traj_rel, decoder_h):
        """
        Assuming obs_len = 20, pred_len = 30:

        traj_abs (20, b, 2)
        traj_rel (20, b, 2)
        decoder_h: (1, b, self.h_dim)  
        """

        num_agents = traj_abs.size(1)

        # Adapt decoder hidden state input to number of layers and bidirectional 

        decoder_h = decoder_h.repeat_interleave(repeats=self.D*self.num_layers,dim=0) # self.D*self.num_layers x b x self.h_dim
        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda(self.current_cuda)
        state_tuple = (decoder_h, decoder_c)
        
        if self.use_rel_disp:
            pred_traj_fake_rel = []
            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(num_agents, -1))))
            decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)
  
            for _ in range(self.pred_len):
                output, state_tuple = self.decoder(decoder_input, state_tuple)
    
                if self.D == 2: # LSTM bidirectional
                    output = output[:,:,self.h_dim:] # Take the forward information from the last stacked LSTM layer
                                                    # state, not the reverse

                                                    # L0(F->R) -> L1(F->R) -> L2(F->R) ...

                rel_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim)))
                traj_rel = torch.roll(traj_rel, -1, dims=(0))
                traj_rel[-1] = rel_pos
                pred_traj_fake_rel.append(rel_pos.contiguous().view(num_agents,-1))

                decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(num_agents, -1))))
                decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)
            
            pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)

            return pred_traj_fake_rel
        else:
            # TODO: How to generate the absolute positions directly without using rel-rel?

            pred_traj_fake = []

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_abs.contiguous().view(num_agents, -1))))
            decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)

            for _ in range(self.pred_len):
                output, state_tuple = self.decoder(decoder_input, state_tuple)
                abs_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim)))
                pred_traj_fake.append(abs_pos.contiguous().view(num_agents,-1))

                traj_abs = torch.roll(traj_abs, -1, dims=(0))
                traj_abs[-1] = abs_pos

                decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_abs.contiguous().view(num_agents, -1))))
                decoder_input = decoder_input.contiguous().view(1, num_agents, self.embedding_dim)

            pred_traj_fake = torch.stack(pred_traj_fake, dim=0)

            return pred_traj_fake

class MM_DecoderLSTM(nn.Module):

    def __init__(self, input_len=20, output_len=30, h_dim=64, embedding_dim=16, num_modes=6, 
                 num_layers=1, bidirectional=False, dropout=0.5, current_cuda="cuda:0",
                 use_rel_disp=False):
        super().__init__()

        self.pred_len = output_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_modes = num_modes

        traj_points = self.num_modes*2
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)

        self.spatial_embedding = nn.Linear(traj_points, self.embedding_dim) # Last obs * 2 points
        self.hidden2pos = nn.Linear(self.h_dim, traj_points)

        self.confidences = nn.Linear(self.h_dim, self.n_samples)

    def forward(self, traj_abs, traj_rel, state_tuple):
        """
        traj_abs (1, b, 2)
        traj_rel (1, b, 2)
        state_tuple: h and c
            h : c : (1, b, self.h_dim)
        """

        t, batch_size, f = traj_abs.shape
        traj_rel = traj_rel.view(t,batch_size,1,f)
        traj_rel = traj_rel.repeat_interleave(self.n_samples, dim=2)
        traj_rel = traj_rel.view(t, batch_size, -1)

        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel.contiguous().view(batch_size, -1))) # bx16
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim) # 1 x batch x 16
        for _ in range(self.pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) # output (1, b, 32)

            rel_pos = self.hidden2pos(state_tuple[0].contiguous().view(-1, self.h_dim)) #(b, 2*m)

            decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.seq_len, batch_size, self.n_samples, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) #(b, m, 30, 2)
        conf = self.confidences(state_tuple[0].contiguous().view(-1, self.h_dim))
        conf = torch.softmax(conf, dim=1)

        pdb.set_trace()
        
        return pred_traj_fake_rel, conf

class GoalDecoderLSTM(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim) # Last obs * 2 points
        self.hidden2pos = nn.Linear(3*self.h_dim, 2)

        self.goal_embedding = nn.Linear(2*32,self.h_dim)
        self.abs_embedding = nn.Linear(2,self.h_dim)

    def forward(self, traj_abs, traj_rel, state_tuple, goals):
        """
            traj_abs (1, b, 2)
            traj_rel (1, b, 2)
            state_tuple: h and c
                h : c : (1, b, self.h_dim)
            goals: b x 32 x 2
        """

        batch_size = traj_abs.size(1)

        goals = goals.view(goals.shape[0],-1) 
        goals_embedding = self.goal_embedding(goals) # b x h_dim

        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel.contiguous().view(batch_size, -1))) # bx16
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim) # 1 x batch x 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) # output (1, b, 32)
            traj_abs_embedding = self.abs_embedding(traj_abs.view(traj_abs.shape[1],-1)) # b x h_dim

            input_final = torch.cat((state_tuple[0],
                                     traj_abs_embedding.unsqueeze(0),
                                     goals_embedding.unsqueeze(0)),dim=2) # 1 x b x 3*h_dim

            rel_pos = self.hidden2pos(input_final.view(input_final.shape[1],-1)) # (b, 2)
            decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

            traj_abs = traj_abs + rel_pos # Update next absolute point

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class MM_GoalDecoderLSTM(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16, n_samples=3):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.n_samples = n_samples
        
        traj_points = self.n_samples*2
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(traj_points, self.embedding_dim) # Last obs * 2 points
        self.hidden2pos = nn.Linear(2*self.h_dim, traj_points)
        self.confidences = nn.Linear(self.h_dim, self.n_samples)

        self.goal_embedding = nn.Linear(2*32,self.h_dim)

    def forward(self, traj_abs, traj_rel, state_tuple, goals):
        """
            traj_abs (1, b, 2)
            traj_rel (1, b, 2)
            state_tuple: h and c
                h : c : (1, b, self.h_dim)
            goals: b x 32 x 2
        """

        t, batch_size, f = traj_abs.shape
        traj_rel = traj_rel.view(t,batch_size,1,f)
        traj_rel = traj_rel.repeat_interleave(self.n_samples, dim=2)
        traj_rel = traj_rel.view(t, batch_size, -1)

        goals = goals.view(goals.shape[0],-1) 
        goals_embedding = self.goal_embedding(goals) # b x h_dim

        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel.contiguous().view(batch_size, -1))) # bx16
        decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim) # 1 x batch x 16
        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) # output (1, b, 32)

            input_final = torch.cat((state_tuple[0],
                                     goals_embedding.unsqueeze(0)),dim=2)

            rel_pos = self.hidden2pos(input_final.view(input_final.shape[1],-1)) #(b, 2*m)

            decoder_input = F.leaky_relu(self.spatial_embedding(rel_pos.contiguous().view(batch_size, -1)))
            decoder_input = decoder_input.contiguous().view(1, batch_size, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch_size,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pdb.set_trace()
        pred_traj_fake_rel = pred_traj_fake_rel.view(self.seq_len, batch_size, self.n_samples, -1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,2,0,3) #(b, m, 30, 2)
        conf = self.confidences(state_tuple[0].contiguous().view(-1, self.h_dim))
        conf = torch.softmax(conf, dim=1)
        return pred_traj_fake_rel, conf

class BaseDecoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.
    Defined in :numref:`sec_encoder-decoder`"""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError