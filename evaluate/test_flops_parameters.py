#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Test performance of the models in terms of FLOPs (MACs) and parameters

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import time
import pdb
import yaml
import os
from prodict import Prodict

# DL & Math

import torch
from thop import profile, clever_format
# https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
from fvcore.nn import FlopCountAnalysis, flop_count_table

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

import model.utils.utils as utils

from model.models.social_lstm_mhsa_old import TrajectoryGenerator as TG_So_LSTM_MHSA
from model.models.social_set_transformer_mm import TrajectoryGenerator as TG_So_SET_Trans_MM

#######################################

current_cuda = torch.device(f"cuda:1")
device = torch.device(current_cuda if torch.cuda.is_available() else "cpu")

# Get model parameters and FLOPs (Floating Point Operation per second)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
agents = 20

## Social LSTM MHSA

aux_path = "save/argoverse/social_lstm_mhsa/best_unimodal_100_percent/config_file.yml"
config_path = os.path.join(BASE_DIR,aux_path)
with open(config_path) as config_file:
    config_file = yaml.safe_load(config_file)
    config_file = Prodict.from_dict(config_file)

h_dim = 32 # LSTM hidden state
print("Social LSTM MHSA: ")

print("Num agents: ", agents)

m_train = TG_So_LSTM_MHSA(config_encoder_lstm=config_file.model.generator.encoder_lstm,
                          config_decoder_lstm=config_file.model.generator.decoder_lstm,
                          config_mhsa=config_file.model.generator.mhsa,
                          current_cuda=current_cuda).train()

m_non_train = TG_So_LSTM_MHSA(config_encoder_lstm=config_file.model.generator.encoder_lstm,
                          config_decoder_lstm=config_file.model.generator.decoder_lstm,
                          config_mhsa=config_file.model.generator.mhsa,
                          current_cuda=current_cuda)

print("Only trainable parameters: ", utils.count_parameters(m_train))
print("All parameters: ", utils.count_parameters(m_non_train))

obs = torch.randn(20,agents,2).to(device)
rel = torch.randn(20,agents,2).to(device)
se = torch.tensor([[0,agents]]).to(device)
idx = torch.tensor([1]).to(device)

model = m_non_train.to(device)

t0 = time.time()

preds = model(obs,rel,se,idx) # forward
print("time ", time.time() - t0)

print("FLOPs count using thop library: ")

macs, params = profile(model, inputs=(obs,rel,se,idx, ), custom_ops={})
macs, params = clever_format([macs, params], "%.3f")

print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print("MACs count using fvcore library: ")

macs = FlopCountAnalysis(model,(obs,rel,se,idx)) # This function actually 
# returns the MACs. https://github.com/facebookresearch/fvcore/issues/69

print("MACs total: ", macs.total())
print("MACs my module: ", macs.by_module())
print("MACs by module and operator: ", macs.by_module_and_operator())

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

## Social SET Transformer Multimodal

# agents = 20

# print("Social SET Transformer MM: ")

# print("\nNum agents: ", agents)

# m_train = TG_So_SET_Trans_MM().train()
# m_non_train = TG_So_SET_Trans_MM()
# print("Only trainable parameters: ", utils.count_parameters(m_train))
# print("All parameters: ", utils.count_parameters(m_non_train))

# x = torch.randn(20,agents,2).to(device)
# seq_start_end = torch.tensor([[0,agents]]).to(device)

# model = m_non_train.to(device)

# t0 = time.time()
# preds, conf = model(x, seq_start_end) # forward
# print("time ", time.time() - t0)

# print("MACs count using thop library: ")

# macs, params = profile(model, inputs=(x, seq_start_end, ), custom_ops={})
# macs, params = clever_format([macs, params], "%.3f")

# print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# print("MACs count using fvcore library: ")

# agents_list = [1,10,20,50,100,500,1000] # Test the FLOPs for this number of agents
# flops_list = []

# for num_agents in agents_list:
#     x = torch.randn(20,num_agents,2).to(device)
#     seq_start_end = torch.tensor([[0,num_agents]]).to(device)
#     macs = FlopCountAnalysis(model,(x,seq_start_end))
#     flops = 0.5 * macs.total()
#     flops = clever_format([flops], "%.3f") 
    
#     flops_list.append(flops)

# print("SET Transformer FLOPs study: \n")
# for num_agents, flops in zip(agents_list, flops_list):
#     print(f"Agents: {num_agents}, FLOPs: {flops}")

# print("MACs total: ", macs.total())
# print("MACs my module: ", macs.by_module())
# print("MACs by module and operator: ", macs.by_module_and_operator())

# print(flop_count_table(macs))

################################################################

# LSTM 10 agents: 46616 (totally wrong)
# Set transformer 10 agents: 378528 (makes sense)

# It is not possible to compute LSTM for just one agent


