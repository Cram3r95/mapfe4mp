import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import pdb

# Variation of the PV_LSTM network with position decoder
class PV_LSTM(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, pred_len=30, current_cuda='cuda:0'):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.current_cuda = current_cuda
        
        super(PV_LSTM, self).__init__()
        
        self.speed_encoder = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim)
        self.pos_encoder   = nn.LSTM(input_size=self.data_dim, hidden_size=self.hidden_dim)
        
        self.pos_embedding = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.data_dim),
                                           nn.ReLU())
        
        self.speed_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=self.hidden_dim)
        self.pos_decoder    = nn.LSTMCell(input_size=self.data_dim, hidden_size=self.hidden_dim)
        self.fc_speed    = nn.Linear(in_features=self.hidden_dim, out_features=self.data_dim)
        
    def forward(self, pos, speed, seq_start_end):
        """

        In Argoverse:
        
        pos = obs_traj (20 x bs x 2). Absolute positions (around 0,0)
        speed = obs_traj_rel (20 x bs x 2). Relative displacements = "velocity"
        
        Only for the target agent
        """

        _, (hpo, cpo) = self.pos_encoder(pos)
        hpo = hpo.squeeze(0)
        cpo = cpo.squeeze(0)

        _, (hsp, csp) = self.speed_encoder(speed)
        hsp = hsp.squeeze(0)
        csp = csp.squeeze(0)
        
        speed_outputs    = torch.tensor([], device=self.current_cuda)
        pos_outputs    = torch.tensor([], device=self.current_cuda)
        in_sp = speed[-1,:,:] # Last relative displacement ("velocity")
        in_pos = pos[-1,:,:] # Last position
        
        # Concat features

        hds = hpo + hsp # hidden state
        cds = cpo + csp # cell state

        for _ in range(self.pred_len):
            hds, cds         = self.speed_decoder(in_sp, (hds, cds))
            speed_output     = self.fc_speed(hds)
            speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(0)), dim = 0)
            
            hpo, cpo         = self.pos_decoder(in_pos, (hpo, cpo))
            pos_output       = self.fc_speed(hpo)
            pos_outputs      = torch.cat((pos_outputs, pos_output.unsqueeze(0)), dim = 0)

        return pos_outputs, speed_outputs