import torch
from torch import nn
from prodict import Prodict
import torch.nn.functional as F

from model.modules.layers import MLP

class EncoderLSTM(nn.Module):

    def __init__(self, embedding_dim=16, h_dim=64):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        h = torch.zeros(1,batch, self.h_dim).cuda()
        c = torch.zeros(1,batch, self.h_dim).cuda()
        return h, c

    def forward(self, obs_traj):

        npeds = obs_traj.size(1)

        obs_traj_embedding = F.leaky_relu(self.spatial_embedding(obs_traj.contiguous().view(-1, 2)))
        obs_traj_embedding = obs_traj_embedding.view(-1, npeds, self.embedding_dim)
        state = self.init_hidden(npeds)
        output, state = self.encoder(obs_traj_embedding, state)
        final_h = state[0]
        final_h = final_h.view(npeds, self.h_dim)
        return final_h

class BaseEncoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError