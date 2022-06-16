import yaml
import numpy as np
from prodict import Prodict
import pdb
import sys
import os

import matplotlib.pyplot as plt

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

import torch

from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset, seq_collate
from torch.utils.data import DataLoader
from model.models.social_set_transformer_mm import TrajectoryGenerator as TrajectoryGenerator_SocialSetTransMM
import model.datasets.argoverse.map_functions as map_functions                                        
from argoverse.map_representation.map_api import ArgoverseMap

avm = ArgoverseMap()

pred_len = 30

# Load config

config_path = BASE_DIR + '/config/config_social_set_transformer_mm.yml'

with open(config_path) as config:
        config = yaml.safe_load(config)
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR

# Fill some additional dimensions

past_observations = config.hyperparameters.obs_len

config.dataset.split = "val"
config.dataset.split_percentage = 0.55 # To generate the final results, must be 1 (whole split test) 0.0001
config.dataset.start_from_percentage = 0.0
config.dataset.batch_size = 1 # Better to build the h5 results file
config.dataset.num_workers = 0
config.dataset.class_balance = -1.0 # Do not consider class balance in the split val
config.dataset.shuffle = False

config.hyperparameters.pred_len = 30 # In test, we do not have the gt (prediction points)

data_images_folder = BASE_DIR + "/" + config.dataset.path + config.dataset.split + "/data_images"

data_val = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                    root_folder=config.dataset.path,
                                                    obs_len=config.hyperparameters.obs_len,
                                                    pred_len=config.hyperparameters.pred_len,
                                                    distance_threshold=config.hyperparameters.distance_threshold,
                                                    split=config.dataset.split,
                                                    split_percentage=config.dataset.split_percentage,
                                                    start_from_percentage=config.dataset.start_from_percentage,
                                                    shuffle=False,
                                                    batch_size=config.dataset.batch_size,
                                                    class_balance=config.dataset.class_balance,
                                                    obs_origin=config.hyperparameters.obs_origin)

loader = DataLoader(data_val,
                    batch_size=config.dataset.batch_size,
                    shuffle=False,
                    num_workers=config.dataset.num_workers,
                    collate_fn=seq_collate)

exp_name = "social_set_transformer_mm/exp2/"
model_path = BASE_DIR + "/save/argoverse/" + exp_name + "argoverse_motion_forecasting_dataset_0_with_model.pt"
checkpoint = torch.load(model_path)
generator = TrajectoryGenerator_SocialSetTransMM()

print("Loading model ...")
generator.load_state_dict(checkpoint.config_cp['g_best_state'])
generator.cuda() # Use GPU
generator.eval()

PLOT_QUALITATIVE_RESULTS = True
dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

with torch.no_grad():
    for batch_index, batch in enumerate(loader):
        batch = [tensor.cuda() for tensor in batch]
        
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
        loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq,_) = batch

        predicted_traj = []
        pred_traj_fake_list = []

        print("Analizing sequence: ", num_seq.item())

        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        agent_traj_gt = pred_traj_gt[:,agent_idx,:] # From Multi to Single (30 x bs x 2)

        if PLOT_QUALITATIVE_RESULTS:
            filename = data_images_folder + "/" + str(num_seq.item()) + ".png"
            img = map_functions.plot_qualitative_results_mm(filename, pred_traj_fake_list, agent_traj_gt, agent_idx,
                                                        object_cls, obs_traj, ego_origin, dist_rasterized_map)

