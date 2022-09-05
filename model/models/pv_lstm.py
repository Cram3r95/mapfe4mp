## STABLE

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import torchvision.transforms as transforms

# import pdb

# # Variation of the PV_LSTM network with position decoder
# class TrajectoryGenerator(nn.Module):
#     def __init__(self, data_dim=2, hidden_dim=256, pred_len=30, current_cuda='cuda:0'):
#         self.data_dim = data_dim
#         self.hidden_dim = hidden_dim
#         self.pred_len = pred_len
#         self.current_cuda = current_cuda
        
#         super(TrajectoryGenerator, self).__init__()
        
#         self.speed_encoder = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim)
#         self.pos_encoder   = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim)
        
#         self.pos_embedding = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.data_dim),
#                                            nn.ReLU())
        
#         self.speed_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=self.hidden_dim)
#         self.pos_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=self.hidden_dim)
#         self.fc_speed    = nn.Linear(in_features=self.hidden_dim, out_features=self.data_dim)
        
#     def forward(self, pos, speed):
#         """

#         In Argoverse:
        
#         pos = obs_traj (20 x bs x 2). Absolute positions (around 0,0)
#         speed = obs_traj_rel (20 x bs x 2). Relative displacements = "velocity"
        
#         Only for the target agent
#         """

#         _, (hpo, cpo) = self.pos_encoder(pos)
#         hpo = hpo.squeeze(0)
#         cpo = cpo.squeeze(0)

#         _, (hsp, csp) = self.speed_encoder(speed)
#         hsp = hsp.squeeze(0)
#         csp = csp.squeeze(0)
        
#         speed_outputs    = torch.tensor([], device=self.current_cuda)
#         pos_outputs    = torch.tensor([], device=self.current_cuda)
#         in_sp = speed[-1,:,:] # Last relative displacement ("velocity")
#         in_pos = pos[-1,:,:] # Last position
        
#         # Concat features

#         hds = hpo + hsp # hidden state
#         cds = cpo + csp # cell state

#         for _ in range(self.pred_len):
#             hds, cds         = self.speed_decoder(in_sp, (hds, cds))
#             speed_output     = self.fc_speed(hds)
#             speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(0)), dim = 0)
            
#             hpo, cpo         = self.pos_decoder(in_pos, (hpo, cpo))
#             pos_output       = self.fc_speed(hpo)
#             pos_outputs      = torch.cat((pos_outputs, pos_output.unsqueeze(0)), dim = 0)

#         return pos_outputs, speed_outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import pdb

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        # layers.append(nn.LeakyReLU())
        layers.append(nn.SELU())
    return nn.Sequential(*layers)

import pdb

# Variation of the PV_LSTM network with position decoder
class TrajectoryGenerator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, pred_len=30, 
                       dropout=0.2, concat_features=False, current_cuda='cuda:0'):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.dropout = dropout
        self.current_cuda = current_cuda
        self.concat_features = concat_features
        
        super(TrajectoryGenerator, self).__init__()
        
        self.speed_encoder = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim, dropout=self.dropout)
        self.pos_encoder   = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim, dropout=self.dropout)

        self.lne = nn.LayerNorm(self.hidden_dim) # Layer normalization encoder

        if self.concat_features:
            decoder_hidden_features = 2*self.hidden_dim
        else:
            decoder_hidden_features = self.hidden_dim

        self.speed_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=decoder_hidden_features)
        self.pos_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=decoder_hidden_features)

        # self.speed_decoder = nn.LSTM(input_size=self.data_dim, hidden_size=decoder_hidden_features, dropout=self.dropout)
        # self.pos_decoder   = nn.LSTM(input_size=self.data_dim, hidden_size=decoder_hidden_features, dropout=self.dropout)

        self.fc_speed    = nn.Linear(in_features=decoder_hidden_features, out_features=self.data_dim)

        mlp_dims = [self.hidden_dim,64,32,2]
        self.mlp = make_mlp(mlp_dims)
        
    def forward(self, pos, speed, seq_start_pos=None, agent_idx=None):
        """

        In Argoverse:
        
        pos = obs_traj (20 x bs x 2). Absolute positions (around 0,0)
        speed = obs_traj_rel (20 x bs x 2). Relative displacements = "velocity"
        
        Only for the target agent
        """

        _, (hpo, cpo) = self.pos_encoder(pos)
        hpo = hpo.squeeze(0)
        cpo = cpo.squeeze(0)

        hpo = self.lne(hpo)
        cpo = self.lne(cpo)

        _, (hsp, csp) = self.speed_encoder(speed)
        hsp = hsp.squeeze(0)
        csp = csp.squeeze(0)

        hsp = self.lne(hsp)
        csp = self.lne(csp)
        
        speed_outputs    = torch.tensor([], device=self.current_cuda)
        pos_outputs    = torch.tensor([], device=self.current_cuda)
        curr_speed = speed[-1,:,:] # Last relative displacement ("velocity")
        curr_pos = pos[-1,:,:] # Last position
        
        # Concat features

        if self.concat_features:
            hds = torch.cat([hpo.contiguous().view(-1,self.hidden_dim),
                             hsp.contiguous().view(-1,self.hidden_dim)],dim=1)
            cds = torch.cat([cpo.contiguous().view(-1,self.hidden_dim),
                             csp.contiguous().view(-1,self.hidden_dim)],dim=1)
        else:
            hds = hpo + hsp # hidden state
            cds = cpo + csp # cell state

        for _ in range(self.pred_len):
            hds, cds         = self.speed_decoder(curr_speed, (hds, cds))
            curr_speed     = self.fc_speed(hds)
            # curr_speed = speed_output
            # curr_speed = self.mlp(hds)
            speed_outputs    = torch.cat((speed_outputs, curr_speed.unsqueeze(0)), dim = 0)
            
            # hpo, cpo         = self.pos_decoder(curr_pos, (hpo, cpo))
            hpo, cpo         = self.pos_decoder(curr_pos, (hpo, cpo))
            curr_pos       = self.fc_speed(hpo)
            # curr_pos = pos_output
            # curr_pos = self.mlp(hpo)
            pos_outputs      = torch.cat((pos_outputs, curr_pos.unsqueeze(0)), dim = 0)

            # hds = hpo + hsp # hidden state
            # cds = cpo + csp # cell state

        return pos_outputs, speed_outputs