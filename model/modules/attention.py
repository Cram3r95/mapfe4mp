import torch
import math
from torch import nn
import torch.nn.functional as F
import pdb

from model.modules.encoders import BaseEncoder
from model.modules.decoders import BaseDecoder
from model.modules.backbones import CNN

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.
    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 1x80x10x32

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3) # 1x10x80x32

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3]) # 

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.
    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    Defined in :numref:`sec_utils`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

class SATAttentionModule(nn.Module):
    
    def __init__(self, config):
        super(SATAttentionModule, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.linear_decoder = nn.Linear(
            self.config.linear_decoder.in_features, self.config.linear_decoder.out_features
        )
        self.linear_feature = nn.Linear(
            self.config.linear_feature.in_features, self.config.linear_feature.out_features
        )
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax(self.config.softmax.dim)

    def forward(self, feature_1, feature_decoder, num_agents=32):
        """
        Inputs:
            - feature_1: Visual Extracto8r output (4D {batch*1, 512, 18, 18}) or Joint Extractor output (3D {L, 32*batch, dim_features})
            - feature_decoder: Generator output (LSTM based decoder; 3D {1, num_agents*batch, dim_features})
        Outputs:
            - alpha?
            - context_vector: (For physical attention, concentrates on feasible paths for each agent; For Social attention, highlights
              which other agents are most important to focus on when predicting the trajectory of the agent i)
        """

        # Feature decode processing

        batch = feature_decoder.size(1) # num_agents x batch_size
        batch_size = int(batch/num_agents)

        for i in range(batch_size):
            feature_decoder_ind = feature_decoder[:,num_agents*i:num_agents*(i+1),:] 
            feature_decoder_ind = feature_decoder_ind.contiguous().view(-1,feature_decoder.size(2))
            if (len(feature_1.size()) == 4):
                # Visual Extractor

                feature_1_ind = torch.unsqueeze(feature_1[i, :, :, :],0)
                feature_1_ind = feature_1_ind.contiguous().view(-1,feature_1_ind.size(2)*feature_1_ind.size(3)) # 4D -> 2D
            elif (len(feature_1.size()) == 3):
                # Joint Extractor


                feature_1_ind = feature_1[:,num_agents*i:num_agents*(i+1),:]
                feature_1_ind = feature_1_ind.contiguous().view(-1, num_agents) # 3D -> 2D
                # feature_1_ind = feature_1_ind.contiguous().view(batch, -1) # 3D -> 2D
            
            linear_feature1_output = self.linear_feature(feature_1_ind)
            
            # Feature decoder processing
            linear_decoder_output = self.linear_decoder(feature_decoder_ind)
        
            alpha = self.softmax(linear_decoder_output) # 32 x 512

            if i == 0:
                list_context_vector = torch.matmul(alpha, linear_feature1_output)
            else:
                context_vector = torch.matmul(alpha, linear_feature1_output)
                list_context_vector = torch.cat((list_context_vector,context_vector), 0)
        
        return alpha, list_context_vector


####################################################################
####################### Transformer  ###############################
####################################################################

class PositionalEncoding(nn.Module):
    """
    Positional encoding (absolute)
        input: X
        output: X + P
        P: positional embedding matrix
    """

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1,1) / torch.pow(
            10000, torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens
        )
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class DotProductAttention(nn.Module):
    """
    Scaled dot product attention
        a(q,k) = q*k/sqrt(d)
        Computationally more effient than Additive. This attention requires "d" to keep the variance
        in q and k. Means its not modified.
    """

    def __init__(self, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Shape of `queries`: (`batch_size`, no. of queries, `d`)
        Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
        Shape of `values`: (`batch_size`, no. of key-value pairs, value
            dimension)
        Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
        Multi-head attention combines knowledge of the same attention pooling via different 
        representation subspaces of queries, keys, and values.
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        #TODO select attention mechanism: additive or dotproduct
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        Shape of `queries`, `keys`, or `values`:
            (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        Shape of `valid_lens`:
            (`batch_size`,) or (`batch_size`, no. of queries)
        After transposing, shape of output `queries`, `keys`, or `values`:
            (`batch_size` * `num_heads`, no. of queries or key-value pairs,
            `num_hiddens` / `num_heads`)
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class AddNorm(nn.Module):
    """"
    Residual connection followed by layer normalization
        Layer normalization is the same as batch normalization except that the
        former normalizes across the feature dimension. Despite its pervasive applications in computer
        vision, batch normalization is usually empirically less effective than layer normalization in 
        natural language processing tasks, whose inputs are often variable-length sequences.
    """

    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        #TODO select normalization: layernorm or batch normalization
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """
    Positionwise feed-forward network.
        The positionwise feed-forward network transforms the representation at all 
        the sequence positions using the same MLP
    """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """"
        output: (batch size, number of time steps, ffn_num_outputs)
        """
        return self.dense2(self.relu(self.dense1(X)))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block.
        - Self-Attention Multi-Head Attention
        - AddNorm
        - PositionwiseFFN
        - AddNorm
    """

    def __init__(
        self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, 
        ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder
        - Embedding
        - Positional encoding
        - EncoderBlock
    """

    def __init__(self, traj_size, key_size, query_size, value_size, num_hiddens, norm_shape, 
        ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, 
        use_bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(traj_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i),
                EncoderBlock(
                    key_size, query_size, value_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias
                )
            )

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X


class DecoderBlock(nn.Module):
    """
    Transformer decoder block.
        - Masked Self-Attention Multi-Head Attention
        - AddNorm
        - Multi-Head Attention (encoder hidden states as key and value)
        - AddNorm
        - PositionwiseFFN
        - AddNorm
    """

    def __init__(
        self, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        dropout, i, **kwargs
    ):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens,num_hiddens
        )
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(BaseDecoder):

    def __init__(self, traj_size, key_size, query_size, value_size,
        num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
        num_heads, num_layers, dropout, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Linear(traj_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, i)
            )
        self.dense = nn.Linear(num_hiddens, num_hiddens)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        # encoding
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            # blocks -> each block is decoder transformer
            X, state = blk(X, state)
        return self.dense(X), state