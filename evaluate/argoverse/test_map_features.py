#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Test map features extraction

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import time
import time
import pandas as pd
import os
import git

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.dataset_utils as dataset_utils

from model.datasets.argoverse.map_functions import MapFeaturesUtils
from argoverse.map_representation.map_api import ArgoverseMap

# Global variables

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

map_features_utils_instance = MapFeaturesUtils()
avm = ArgoverseMap()

obs_len = 20
pred_len = 30

splits = ["train","val","test"]
modes = ["train","test"] # if train -> compute the best candidate (oracle)
                         # Otherwise, return N plausible candidates
limit = 150

for split in splits:
    print("Evaluating: ", split)
    folder = os.path.join(BASE_DIR,"data/datasets/argoverse/motion-forecasting",split,"data")
    files, num_files = dataset_utils.load_list_from_folder(folder)
    file_id_list, root_file_name = dataset_utils.get_sorted_file_id_list(files)
    
    for i,seq_id in enumerate(file_id_list):
        if i+1 >= limit:
            break
        print("Seq_id: ", seq_id)
        seq_path = f"{BASE_DIR}/data/datasets/argoverse/motion-forecasting/{split}/data/{seq_id}.csv"
        df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

        # Get social and map features for the agent

        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

        for mode in modes:
            # Map features extraction

            start = time.time()
            map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
                    agent_track,
                    seq_id,
                    split,
                    obs_len,
                    obs_len + pred_len,
                    RAW_DATA_FORMAT,
                    mode,
                    avm
                )
            end = time.time()
            print("Time consumed: ", end-start)

# pdb.set_trace()