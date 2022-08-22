from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torchvision.transforms.functional as TF

from model.modules.attention import MultiHeadAttention
from model.modules.encoders_old import Old_EncoderLSTM as Encoder
from model.modules.decoders import Old_TemporalDecoderLSTM as TemporalDecoder

MAX_PEDS = 32

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    return torch.randn(*shape).cuda()


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len=20, pred_len=30, mlp_dim=64, h_dim=32, embedding_dim=16, bottleneck_dim=32,
        noise_dim=8, n_agents=10, img_feature_size=(512,6,6), dropout=0.3
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.bottleneck_dim = bottleneck_dim
        self.noise_dim = noise_dim
        self.n_agents = n_agents

        self.encoder = Encoder(h_dim=self.h_dim)
        self.lne = nn.LayerNorm(self.h_dim)

        self.sattn = MultiHeadAttention(
            key_size=self.h_dim, query_size=self.h_dim, value_size=self.h_dim,
            num_hiddens=self.h_dim, num_heads=4, dropout=dropout
        )

        self.decoder = TemporalDecoder(h_dim=self.h_dim)

        mlp_context_input = self.h_dim*2 # concat of social context and trajectories embedding
        self.lnc = nn.LayerNorm(mlp_context_input)
        mlp_decoder_context_dims = [mlp_context_input, self.mlp_dim, self.h_dim - self.noise_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)

    def add_noise(self, _input):
        npeds = _input.size(0)
        noise_shape = (self.noise_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, -1).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=1)

    def forward(self, obs_traj, obs_traj_rel, start_end_seq, agent_idx=None):
        """
            n: number of objects in all the scenes of the batch
            b: batch
            obs_traj: (20,n,2)
            obs_traj_rel: (20,n,2)
            start_end_seq: (b,2)
            agent_idx: (b, 1) -> index of agent in every sequence.
                None: trajectories for every object in the scene will be generated
                Not None: just trajectories for the agent in the scene will be generated
            -----------------------------------------------------------------------------
            pred_traj_fake_rel:
                (30,n,2) -> if agent_idx is None
                (30,b,2)
        """
        
        ## Encode trajectory
        final_encoder_h = self.encoder(obs_traj_rel) # batchx32
        final_encoder_h = torch.unsqueeze(final_encoder_h, 0) #  1xbatchx32
        # queries -> indican la forma del tensor de salida (primer argumento)
        final_encoder_h = self.lne(final_encoder_h)

        ## Social Attention to encoded trajectories
        attn_s = []
        for start, end in start_end_seq.data:
            attn_s_batch = self.sattn(
                final_encoder_h[:,start:end,:], final_encoder_h[:,start:end,:], final_encoder_h[:,start:end,:], None
            ) # 8x10x32 # multi head self attention
            attn_s.append(attn_s_batch)
        attn_s = torch.cat(attn_s, 1)
        
        ## create decoder context input
        mlp_decoder_context_input = torch.cat(
            [
                final_encoder_h.contiguous().view(-1, 
                self.h_dim), attn_s.contiguous().view(-1, self.h_dim)
            ],
            dim=1
        ) # 80 x (32*3)
        if agent_idx is not None:
            mlp_decoder_context_input = mlp_decoder_context_input[agent_idx,:]

        ## add noise to decoder input
        noise_input = self.mlp_decoder_context(self.lnc(mlp_decoder_context_input)) # 80x24
        decoder_h = self.add_noise(noise_input) # 80x32
        decoder_h = torch.unsqueeze(decoder_h, 0) # 1x80x32

        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda() # 1x80x32
        state_tuple = (decoder_h, decoder_c)

        # Get agent observations
        if agent_idx is not None: # for single agent prediction
            last_pos = obs_traj[:, agent_idx, :]
            last_pos_rel = obs_traj_rel[:, agent_idx, :]
        else:
            last_pos = obs_traj[-1, :, :]
            last_pos_rel = obs_traj_rel[-1, :, :]

        # decode trajectories
        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, mlp_dim=64, h_dim=64):
        super(TrajectoryDiscriminator, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim

        self.encoder = Encoder()
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims)

    def forward(self, traj_rel):

        final_h = self.encoder(traj_rel)
        scores = self.real_classifier(final_h)
        return scores