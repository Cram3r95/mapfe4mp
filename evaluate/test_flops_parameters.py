#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Preprocess the raw data. Get .npy files in order to avoid processing
## the trajectories and map information each time

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import time

# DL & Math

import torch
import torchstat
from thop import profile, clever_format
from pthflops import count_ops

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

import model.utils.utils as utils

from model.models.social_lstm_mhsa import TrajectoryGenerator as TG_So_LSTM_MHSA
from model.models.social_set_transformer_mm import TrajectoryGenerator as TG_So_SET_Trans_MM

#######################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get model parameters and FLOPs (Floating Point Operation per second)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

## Social LSTM MHSA

h_dim = 32 # LSTM hidden state
print("Social LSTM MHSA h_dim %i: " % h_dim)

m_train = TG_So_LSTM_MHSA(h_dim=h_dim).train()
m_non_train = TG_So_LSTM_MHSA(h_dim=h_dim)
print("Only trainable parameters: ", utils.count_parameters(m_train))
print("All parameters: ", utils.count_parameters(m_non_train))

obs = torch.randn(20,3,2).to(device)
rel = torch.randn(20,3,2).to(device)
se = torch.tensor([[0,3]]).to(device)
idx = torch.tensor([1]).to(device)

model = m_non_train.to(device)

t0 = time.time()
preds = model(obs,rel,se,idx) # forward
print("time ", time.time() - t0)

macs, params = profile(model, inputs=(obs,rel,se,idx, ), custom_ops={})
macs, params = clever_format([macs, params], "%.3f")

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

## Social SET Transformer Multimodal

print("Social SET Transformer MM: ")

h_dim = 32 # LSTM hidden state
m_train = TG_So_SET_Trans_MM().train()
m_non_train = TG_So_SET_Trans_MM()
print("Only trainable parameters: ", utils.count_parameters(m_train))
print("All parameters: ", utils.count_parameters(m_non_train))

x = torch.randn(20,10,2).to(device)
seq_start_end = torch.tensor([[0,5], [5, 10]]).to(device)

model = m_non_train.to(device)

t0 = time.time()
preds, conf = model(x, seq_start_end) # forward
print("time ", time.time() - t0)

macs, params = profile(model, inputs=(x, seq_start_end, ), custom_ops={})
macs, params = clever_format([macs, params], "%.3f")

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# count_ops(model, (x,seq_start_end))

def test_count_flops():
    """
    """

    from torchvision.models import resnet18

    # Create a network and a corresponding input
    device = 'cuda:0'
    model = resnet18().to(device)
    inp = torch.rand(1,3,224,224).to(device)

    # Count the number of FLOPs
    count_ops(model, inp)


