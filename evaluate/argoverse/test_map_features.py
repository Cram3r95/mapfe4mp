#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Test map features extraction

"""
Created on Wed May 18 17:59:06 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import sys
import time
import time
import pandas as pd

# Custom imports

BASE_DIR = "/home/denso/carlos_vsr_workspace/mapfe4mp"
sys.path.append(BASE_DIR)

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

obs_len = 20
pred_len = 30
mode = "test"

split = "val"
seq = "4723"

seq_path = f"/home/denso/carlos_vsr_workspace/mapfe4mp/data/datasets/argoverse/motion-forecasting/{split}/data/{seq}.csv"
df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

# Get social and map features for the agent
agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

# Map features extraction

avm = ArgoverseMap()

start = time.time()

map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
        agent_track,
        obs_len,
        obs_len + pred_len,
        RAW_DATA_FORMAT,
        mode,
        avm
    )

end = time.time()
print("Time consumed: ", end-start)

# pdb.set_trace()