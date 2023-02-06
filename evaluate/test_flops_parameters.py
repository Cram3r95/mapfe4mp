#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Test performance of the models in terms of FLOPs (MACs) and parameters

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import time
import pdb
import yaml
import os
import git
from prodict import Prodict

# DL & Math

import torch
import numpy as np

from torchsummary import summary
from thop import profile, clever_format
# https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
from fvcore.nn import FlopCountAnalysis, flop_count_table

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.utils.utils as utils

# from model.models.mapfe4mp import TrajectoryGenerator
from model.models.cghformer import TrajectoryGenerator

#######################################

current_cuda = torch.device(f"cuda:0")
device = torch.device(current_cuda if torch.cuda.is_available() else "cpu")

# Get model parameters and FLOPs (Floating Point Operation per second)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

whole_model = TrajectoryGenerator(PHYSICAL_CONTEXT="social",CURRENT_DEVICE=device)
whole_model.to(device)
print("Mapfe4mp social. All parameters: ", utils.count_parameters(whole_model))

# whole_model = TrajectoryGenerator(PHYSICAL_CONTEXT="oracle",CURRENT_DEVICE=device)
# print("Mapfe4mp oracle. All parameters: ", utils.count_parameters(whole_model))

# whole_model = TrajectoryGenerator(PHYSICAL_CONTEXT="plausible_centerlines",CURRENT_DEVICE=device)
# print("Mapfe4mp plausible_centerlines. All parameters: ", utils.count_parameters(whole_model))

# whole_model = TrajectoryGenerator(PHYSICAL_CONTEXT="plausible_centerlines+feasible_area",CURRENT_DEVICE=device)
# whole_model.to(device)
# print("Mapfe4mp plausible_centerlines+feasible_area. All parameters: ", utils.count_parameters(whole_model))

past_observations = 20
data_dim = 2
agents = 10
num_centerlines = 3
num_points_centerline = 40

# We assume bs = 1

obs = torch.randn(past_observations,agents,data_dim).to(device)
rel = torch.randn(past_observations,agents,data_dim).to(device)
se = torch.tensor([[0,agents]]).to(device)
agent_idx = np.array([1])
phy_info = torch.tensor([]).to(device)
relevant_centerlines = torch.randn(1,num_centerlines,num_points_centerline,data_dim).to(device)

# macs, params = profile(whole_model, inputs=(obs,rel,se,agent_idx,phy_info,relevant_centerlines), custom_ops={})
# macs, params = clever_format([macs, params], "%.3f")

# print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

agents_list = [1,10,100,500,1000,10000] # Test the FLOPs for this number of agents
flops_list = []

for num_agents in agents_list:
    obs = torch.randn(20,agents,2).to(device)
    rel = torch.randn(20,agents,2).to(device)
    se = torch.tensor([[0,agents]]).to(device)
    # macs = FlopCountAnalysis(whole_model,(obs,rel,se,agent_idx,phy_info,relevant_centerlines))
    macs, params = profile(whole_model, inputs=(obs,rel,se,agent_idx,phy_info,relevant_centerlines), custom_ops={})
    flops = 0.5 * macs
    flops = clever_format([flops], "%.3f") 
    
    flops_list.append(flops)

print("Model FLOPs study: \n")
for num_agents, flops in zip(agents_list, flops_list):
    print(f"Agents: {num_agents}, FLOPs: {flops}")

# print("MACs total: ", macs.total())
# print("MACs my module: ", macs.by_module())
# print("MACs by module and operator: ", macs.by_module_and_operator())
# macs = FlopCountAnalysis(whole_model,(obs,rel,se,agent_idx,phy_info,relevant_centerlines))
# print(flop_count_table(macs))

# ## Social LSTM MHSA

# aux_path = "save/argoverse/social_lstm_mhsa/best_unimodal_100_percent/config_file.yml"
# config_path = os.path.join(BASE_DIR,aux_path)
# with open(config_path) as config_file:
#     config_file = yaml.safe_load(config_file)
#     config_file = Prodict.from_dict(config_file)

# h_dim = 32 # LSTM hidden state
# print("Social LSTM MHSA: ")

# print("Num agents: ", agents)

# m_train = TG_So_LSTM_MHSA(config_encoder_lstm=config_file.model.generator.encoder_lstm,
#                           config_decoder_lstm=config_file.model.generator.decoder_lstm,
#                           config_mhsa=config_file.model.generator.mhsa,
#                           current_cuda=current_cuda).train()

# m_non_train = TG_So_LSTM_MHSA(config_encoder_lstm=config_file.model.generator.encoder_lstm,
#                           config_decoder_lstm=config_file.model.generator.decoder_lstm,
#                           config_mhsa=config_file.model.generator.mhsa,
#                           current_cuda=current_cuda)

# print("Only trainable parameters: ", utils.count_parameters(m_train))
# print("All parameters: ", utils.count_parameters(m_non_train))

# obs = torch.randn(20,agents,2).to(device)
# rel = torch.randn(20,agents,2).to(device)
# se = torch.tensor([[0,agents]]).to(device)
# idx = torch.tensor([1]).to(device)

# model = m_non_train.to(device)

# t0 = time.time()

# preds = model(obs,rel,se,idx) # forward
# print("time ", time.time() - t0)

# print("FLOPs count using thop library: ")

# macs, params = profile(model, inputs=(obs,rel,se,idx, ), custom_ops={})
# macs, params = clever_format([macs, params], "%.3f")

# print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# print("MACs count using fvcore library: ")

# macs = FlopCountAnalysis(model,(obs,rel,se,idx)) # This function actually 
# # returns the MACs. https://github.com/facebookresearch/fvcore/issues/69

# print("MACs total: ", macs.total())
# print("MACs my module: ", macs.by_module())
# print("MACs by module and operator: ", macs.by_module_and_operator())

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

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


