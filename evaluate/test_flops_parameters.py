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
import yaml
from prodict import Prodict

# Plot imports

import matplotlib.pyplot as plt

BASE_DIR = "/home/denso/carlos_vsr_workspace/efficient-goals-motion-prediction"
sys.path.append(BASE_DIR)

# Custom imports

from model.dataset.argoverse.dataset import ArgoverseMotionForecastingDataset

#######################################

def count_models_parameters():
    from sophie.utils.utils import count_parameters
    from sophie.models.mp_sovi import TrajectoryGenerator as TG_SOVI
    from sophie.models.mp_trans_so_set import TrajectoryGenerator as TG_T
    from sophie.models.mp_trans_so_set_goal import TrajectoryGenerator as TG_TSetGoals

    from ptflops import get_model_complexity_info
    from pthflops import count_ops

    from torchsummary import summary

    # obs_traj, obs_traj_rel, seq_start_end, agent_idx

    # obs = torch.randn(20,3,2).cuda()
    # rel = torch.randn(20,3,2).cuda()
    # frames = torch.randn(1,3,224,224).cuda()
    # se = torch.tensor([[0,3]]).cuda()
    # idx = torch.tensor([1]).cuda()

    # with torch.cuda.device(0):
    #     m = TG_SOVI(h_dim=32)
    #     macs, params = get_model_complexity_info(
    #         m, (obs.shape, rel.shape, frames.shape, se.shape, idx.shape), as_strings=True, print_per_layer_stat=True, verbose=True
    #     )
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    obs = torch.randn(20,3,2).cuda().to(device)
    rel = torch.randn(20,3,2).cuda().to(device)
    frames = torch.randn(1,3,224,224).cuda().to(device)
    se = torch.tensor([[0,3]]).cuda().to(device)
    idx = torch.tensor([1]).cuda().to(device)
    model = TG_SOVI().to(device)

    count_ops(model, (obs,rel,frames,se,idx))


    m_train = TG_SOVI(h_dim=32).train()
    m_non_train = TG_SOVI(h_dim=32)
    print("Traj Generator SO 32 Train: ", count_parameters(m_train))
    print("Traj Generator SO 32 Non-Train: ", count_parameters(m_non_train))

    m_train = TG_T().train()
    m_non_train = TG_T()
    print("Traj Generator Trans Train: ", count_parameters(m_train))
    print("Traj Generator Trans Non-Train: ", count_parameters(m_non_train))

    m_train = TG_TSetGoals().train()
    m_non_train = TG_TSetGoals()
    print("Traj Generator Trans Set Goals Train: ", count_parameters(m_train))
    print("Traj Generator Trans Set Goals Non-Train: ", count_parameters(m_non_train))

    a1 = torch.randn(20,10,2).cuda()
    b = torch.tensor([[0,3], [3,6], [6,10]]).cuda()

    # pdb.set_trace()

    # print("Summary SO: ", summary(model=TG_SO, 
    #         input_size=(a1.shape,a1.shape,b.shape), 
    #         batch_size=1))

def test_mmtransformer():
    from sophie.models.mp_mmtrans import mmTrans
    from sophie.utils.utils import count_parameters
    m = mmTrans()
    print("Traj Generator SO 32: ", count_parameters(m))


def test_set_transformer():
    from sophie.models.mp_trans_so_set import TrajectoryGenerator
    from ptflops import get_model_complexity_info
    from pthflops import count_ops

    m = TrajectoryGenerator()
    print(m)
    x = torch.randn(20,10,2)
    se = torch.tensor([[0,5], [5, 10]])
    t0 = time.time()
    preds, conf = m(x, se)
    print("time ", time.time() - t0)

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(
    #         m, (x.shape, se.shape), as_strings=True, print_per_layer_stat=True, verbose=True
    #     )
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    x = torch.randn(20,10,2).to(device)
    se = torch.tensor([[0,5], [5, 10]]).to(device)

    model = m.to(device)
    inp = (x,se)
    # print("input shape: ", inp.shape)
    pdb.set_trace()

    count_ops(model, inp)

    pdb.set_trace()

def test_load_weights():
    from sophie.models.mp_so_goals_decoder import TrajectoryGenerator
    from sophie.utils.utils import load_weights
    import pdb
    m = TrajectoryGenerator()
    print("model ", m)
    w_path = "save/argoverse/gen_exp/exp_multiloss_4/argoverse_motion_forecasting_dataset_0_with_model.pt"
    w = torch.load(w_path)
    

    w_ = load_weights(m, w.config_cp['g_best_state'])
    
    m.load_state_dict(w_)

def test_count_flops():
    """
    """

    import torch
    from torchvision.models import resnet18

    from pthflops import count_ops

    # Create a network and a corresponding input
    device = 'cuda:0'
    model = resnet18().to(device)
    inp = torch.rand(1,3,224,224).to(device)

    # Count the number of FLOPs
    count_ops(model, inp)

def test_count_flops_2():
    """
    """

    from torchvision.models import resnet50
    from thop import profile, clever_format
    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")

    # print("RESNET-50")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>")

    batch_size = 2048
    x = torch.randn(20,batch_size,2)
    se = torch.tensor([[0,5], [5, 10]])

    from sophie.models.mp_trans_so_set import TrajectoryGenerator
    model = TrajectoryGenerator()
    macs, params = profile(model, inputs=(x, se, ))
    macs, params = clever_format([macs, params], "%.3f")
    t0 = time.time()
    print("SET-TRANSFORMER")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("Time consumed by SET-TRANSFORMER: ", time.time()-t0)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

    # x = torch.randn(20,10,2)
    # se = torch.tensor([[0,5], [5, 10]])
    # goals = torch.randn(10,2)

    # from sophie.models.mp_trans_so_set_goal import TrajectoryGenerator as TG_TRANS_GOALS
    # model = TG_TRANS_GOALS()
    # macs, params = profile(model, inputs=(x, se, goals, ))
    # macs, params = clever_format([macs, params], "%.3f")

    # print("SET-TRANSFORMER GOALS")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

    # from sophie.models.mp_so_goals import TrajectoryGenerator as TG
    # m = TG()
    # m.cuda()
    # obs = torch.randn(20,3,2).cuda()
    # rel = torch.randn(20,3,2).cuda()
    # goals = torch.randn(3,32,2).cuda()
    # se = torch.tensor([[0,3]]).cuda()
    # idx = torch.tensor([1]).cuda()

    # macs, params = profile(m, inputs=(obs, rel, goals, se, idx, ))
    # macs, params = clever_format([macs, params], "%.3f")

    # print("LSTM goals")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))



    def forward(self, obs_traj, obs_traj_rel, goal_points, start_end_seq, agent_idx=None):
        """
            n: number of objects in all the scenes of the batch
            b: batch
            obs_traj: (20,b*n,2)
            obs_traj_rel: (20,b*n,2)
            goal_points: (b,32,2)
            start_end_seq: (b,2)
            agent_idx: (b, 1) -> index of agent in every sequence.
                None: trajectories for every object in the scene will be generated
                Not None: just trajectories for the agent in the scene will be generated
            -----------------------------------------------------------------------------
            pred_traj_fake_rel:
                (30,n,2) -> if agent_idx is None
                (30,b,2)
        """