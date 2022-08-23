import cv2
import numpy as np
import os
import copy
import pdb
import sys
import math
import git
import torch
import matplotlib.pyplot as plt
import time

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.dataset as dataset
import model.datasets.argoverse.dataset_utils as dataset_utils                                          

# Load files

dataset_path = "data/datasets/argoverse/motion-forecasting/"
split = "val"
split_folder = BASE_DIR + "/" + dataset_path + split
data_folder = BASE_DIR + "/" + dataset_path + split + "/data/"
data_images_folder = BASE_DIR + "/" + dataset_path + split + "/data_images"

files, num_files = dataset_utils.load_list_from_folder(data_folder)

file_id_list = []
root_file_name = None
for file_name in files:
    if not root_file_name:
        root_file_name = os.path.dirname(os.path.abspath(file_name))
    file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
    file_id_list.append(file_id)
file_id_list.sort()
print("Num files: ", num_files)

file_id = "75" # Override this variable to compute a different local map
dist_around = 40 # m around the ego-vehicle
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]  

path = os.path.join(root_file_name,str(file_id)+".csv")
data = dataset_utils.read_file(path) 

obs_origin = 20
origin_pos, city_name = dataset_utils.get_origin_and_city(data,obs_origin)

obs_len = 20
pred_len = 30
seq_len = obs_len + pred_len
skip = 1
obs_origin = 20

frames = np.unique(data[:, 0]).tolist() 
frame_data = []
for frame in frames:
    frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
idx = 0

num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
curr_seq_rel, id_frame_list, object_class_list, city_id, map_origin = \
    dataset.process_window_sequence(idx, frame_data, frames, \
                                    obs_len, pred_len, file_id, split, obs_origin)

pdb.set_trace()