#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Preprocess the raw data. Get .npy files in order to avoid processing
## the trajectories and map information each time

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import yaml
import time
import os
import git
import pdb
import copy

from prodict import Prodict

# DL & Math

import pandas as pd
import numpy as np
import cv2

from sklearn.metrics.pairwise import euclidean_distances

# Plot imports

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Custom imports

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import centerline_to_polygon

avm = ArgoverseMap()

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from model.datasets.argoverse.map_functions import MapFeaturesUtils
from model.datasets.argoverse.dataset import ArgoverseMotionForecastingDataset
from model.datasets.argoverse.dataset_utils import load_list_from_folder, get_origin_and_city, read_file

#######################################

config_path = os.path.join(BASE_DIR,"config/config_social_lstm_mhsa.yml")
with open(config_path) as config:
    config = yaml.safe_load(config)
    config = Prodict.from_dict(config)
    config.base_dir = BASE_DIR

config.dataset.start_from_percentage = 0.0

# Preprocess data
                         # Split, Process, Split percentage
splits_to_process = dict({"train":[False,1.0], # 0.01 (1 %), 0.1 (10 %), 1.0 (100 %)
                          "val":  [True,1.0],
                          "test": [False,1.0]})
modes_centerlines = ["test"] # "train","test" 
# if train -> compute the best candidate (oracle), only using the "competition" algorithm
# if test, return N plausible candidates. Choose between "competition", "map_api" and "get_around" algorithms

PREPROCESS_SOCIAL_DATA = False
PREPROCESS_RELEVANT_CENTERLINES = True
SAVE_BEV_PLAUSIBLE_AREA = False

obs_len = 20 # steps
pred_len = 30 # steps
freq = 10 # Hz ("steps/s")
obs_origin = 20 
min_dist_around = 25
first_centerline_waypoint = "last_obs" # first_obs, last_obs
max_points = 30 # 30 if start from last_obs, 40 if start from first obs
min_points = 4 # to perform a cubic interpolation you need at least 3 points
algorithm = "map_api"

viz = False
limit_qualitative_results = 150
check_every = 0.1 # % of total files

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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

for split_name,features in splits_to_process.items():
    if features[0]:
        # Agents trajectories

        if PREPROCESS_SOCIAL_DATA:
            print(f"Analyzing social data of {split_name} split ...")
            ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                              root_folder=os.path.join(config.base_dir,config.dataset.path),
                                              obs_len=config.hyperparameters.obs_len,
                                              pred_len=config.hyperparameters.pred_len,
                                              distance_threshold=config.hyperparameters.distance_threshold,
                                              split=split_name,
                                              split_percentage=features[1],
                                              batch_size=config.dataset.batch_size,
                                              class_balance=config.dataset.class_balance,
                                              obs_origin=config.hyperparameters.obs_origin,
                                              preprocess_data=True,
                                              save_data=True)    

        # Most relevant lanes around the target agent

        if PREPROCESS_RELEVANT_CENTERLINES:
            print(f"Analyzing physical information of {split_name} split ...")
            wrong_centerlines = []

            folder = os.path.join(BASE_DIR,config.dataset.path,split_name,"data")
            files, num_files = load_list_from_folder(folder)
            afl = ArgoverseForecastingLoader(folder)

            file_id_list = []
            root_file_name = None
            for file_name in files:
                if not root_file_name:
                    root_file_name = os.path.dirname(os.path.abspath(file_name))
                file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
                file_id_list.append(file_id)
            file_id_list.sort()

            n_files = int(features[1]*num_files)
            print("Num files to analyze: ", n_files)
            start_from = int(config.dataset.start_from_percentage*n_files)

            if (start_from + n_files) >= num_files:
                file_id_list = file_id_list[start_from:]
            else:
                file_id_list = file_id_list[start_from:start_from+n_files]

            ##############################################################################
            # Overwrite file_id_list with some specific sequences to analyze the behaviour
            # E.g. Those sequences for which the interpolated oracle has not been computed 
            # correctly

            # file_id_list = [27353, 57791, 102781, 129915, 143124, 146034, 149913, 154633, 160433, 166298, 168101, 189296, 209572]

            ##############################################################################

            check_every_n_files = int(len(file_id_list)*check_every)
            print(f"Check remaining time every {check_every_n_files} files")
            time_per_iteration = float(0)
            aux_time = float(0)
            
            map_info = dict()
            oracle_centerlines_list = []

            output_dir = os.path.join(BASE_DIR,f"data/datasets/argoverse/motion-forecasting/{split_name}/map_features")
            # output_dir = os.path.join(BASE_DIR,f"data/datasets/argoverse/motion-forecasting/{split_name}/map_features_gray")

            if not os.path.exists(output_dir):
                print("Create trajs folder: ", output_dir)
                os.makedirs(output_dir) # makedirs creates intermediate folders

            for i, file_id in enumerate(file_id_list):
                # print(f"File {file_id} -> {i+1}/{len(file_id_list)}")

                if file_id != -1:

                    if limit_qualitative_results != -1 and i+1 > limit_qualitative_results:
                        viz = False

                    files_remaining = len(file_id_list) - (i+1)
                    seq_path = os.path.join(root_file_name,str(file_id)+".csv")

                    # Compute map features using Argoverse Forecasting Baseline algorithm

                    start = time.time()

                    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

                    # Get social and map features for the agent

                    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
                    city_name = agent_track[0,RAW_DATA_FORMAT["CITY_NAME"]]
                    agent_xy = agent_track[:,[RAW_DATA_FORMAT["X"],RAW_DATA_FORMAT["Y"]]].astype("float")
                    first_obs = agent_xy[0,:]
                    last_obs = agent_xy[obs_origin-1,:]

                    # Filter agent's trajectory (smooth)

                    vel, acc, xy_filtered, extended_xy_filtered = map_features_utils_instance.get_agent_velocity_and_acceleration(agent_xy[:obs_len,:],
                                                                                                                                  filter="least_squares",
                                                                                                                                  debug=False)
                                                                                                        
                    dist_around = vel * (pred_len/freq) + 1/2 * acc * (pred_len/freq)**2

                    if dist_around < min_dist_around:
                        dist_around = min_dist_around

                    # Compute agent's orientation

                    lane_dir_vector, yaw = map_features_utils_instance.get_yaw(xy_filtered, obs_len)

                    for mode in modes_centerlines:
                        # Map features extraction

                        map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
                                agent_track,
                                file_id,
                                split_name,
                                obs_len,
                                obs_len + pred_len,
                                RAW_DATA_FORMAT,
                                mode,
                                avm,
                                viz,
                                algorithm=algorithm # competition, map_api, get_around
                            )

                        if mode == "test": # preprocess N plausible centerlines

                            start_ = time.time()

                            relevant_centerlines = map_feature_helpers["CANDIDATE_CENTERLINES"]  

                            seq_map_info = dict()
                            relevant_centerlines_filtered = []                 

                            for index_centerline, relevant_centerline in enumerate(relevant_centerlines):
                                # Get index of the closest waypoint to the first observation
                                
                                if first_centerline_waypoint == "last_obs":
                                    closest_wp_first, _ = map_features_utils_instance.get_closest_wp(last_obs, relevant_centerline)
                                elif first_centerline_waypoint == "first_obs":
                                    closest_wp_first, _ = map_features_utils_instance.get_closest_wp(first_obs, relevant_centerline)

                                # Get index of the closest waypoint to the last observation + dist_around
                            
                                closest_wp_last, dist_array_last = map_features_utils_instance.get_closest_wp(last_obs, relevant_centerline)
                                dist_array_last = dist_array_last[closest_wp_last:] # To determine dist around from this point

                                idx, value = find_nearest(dist_array_last,dist_around)
                                num_points = idx + (closest_wp_last - closest_wp_first)
                                if num_points < min_points:
                                        num_points = min_points # you must have at least 4 points to conduct a cubic interpolation

                                # Determine if the centerline is inverted (this sometimes happens if the corresponding agent is
                                # carrying a lane change maneuver): The best and most interpretable solution is if the
                                # the end point is closer than the start point and the last observation is closer to the start
                                # rather than the first observation

                                relevant_centerline_orig = copy.deepcopy(relevant_centerline)
                                
                                ec = euclidean_distances(relevant_centerline[:-1,:],relevant_centerline[1:,:])
                                indeces_rows = indeces_cols = np.arange(ec.shape[0])
                                dist = ec[indeces_rows,indeces_cols]

                                invert = False

                                try:
                                    dist_firstobs2start = np.cumsum(dist[:closest_wp_first+1])[-1]
                                    dist_lastobs2start = np.cumsum(dist[:closest_wp_last+1])[-1]
                                    dist_firstobs2end = np.cumsum(dist[closest_wp_first:])[-1]

                                    if ((dist_lastobs2start >= dist_firstobs2start)
                                    and ((closest_wp_first > 0 and closest_wp_first < relevant_centerline.shape[0])
                                        or (dist_firstobs2start <= dist_firstobs2end))):
                                        pass
                                    else:
                                        invert = True

                                except Exception as e:
                                    # print("e: ", e)
                                    invert = True

                                if invert:
                                    # print("invert oracle")

                                    relevant_centerline_orig = copy.deepcopy(relevant_centerline)
                                    relevant_centerline = relevant_centerline[::-1]

                                    # Repeat again the process to obtain the closest wps

                                    # Get index of the closest waypoint to the first observation
                                
                                    closest_wp_first, _ = map_features_utils_instance.get_closest_wp(first_obs, relevant_centerline)

                                    # Get index of the closest waypoint to the last observation + dist_around
                                
                                    closest_wp_last, dist_array_last = map_features_utils_instance.get_closest_wp(last_obs, relevant_centerline)
                                    dist_array_last = dist_array_last[closest_wp_last:]

                                    idx, value = find_nearest(dist_array_last,dist_around) # TODO: Take the next point, or an interpolated point
                                    # between the closest and the next

                                    num_points = idx + (closest_wp_last - closest_wp_first)
                                    if num_points < min_points:
                                        num_points = min_points # you must have at least 4 points to conduct a cubic interpolation

                                # Reduce the lane to the closest N points, starting from the first observation 
                                # (closest to current position) and ending in the closest wp assuming CTRA during pred seconds

                                end_point = agent_xy[-1,:]

                                if closest_wp_first+num_points+1 <= relevant_centerline.shape[0]:
                                    relevant_centerline_filtered = relevant_centerline[closest_wp_first:closest_wp_first+num_points+1,:]
                                else: # If we have reached the end, travel backwards 
                                    # TODO: Interpolate frontwards
                                    back_num_points = (closest_wp_first+num_points+1) - relevant_centerline.shape[0]
                                    # print("back: ", back_num_points)
                                    relevant_centerline_filtered = relevant_centerline[closest_wp_first-back_num_points:,:]
                                
                                if relevant_centerline_filtered.shape[0] != max_points:
                                    try:
                                        # start = time.time()

                                        interpolated_centerline = map_features_utils_instance.interpolate_centerline(relevant_centerline_filtered,
                                                                                                                        max_points=max_points,
                                                                                                                        agent_xy=agent_xy,
                                                                                                                        obs_len=obs_len,
                                                                                                                        seq_len=obs_len+pred_len,
                                                                                                                        split=split_name,
                                                                                                                        seq_id=file_id,
                                                                                                                        viz=viz)

                                        # end = time.time()
                                        # print("Time consumed by interpolation: ", end-start)

                                        assert interpolated_centerline.shape[0] == max_points

                                        relevant_centerlines_filtered.append(interpolated_centerline)

                                        # Store oracle

                                        if index_centerline == 0 and algorithm == "map_api": 
                                            # The first centerline also corresponds to the oracle using map_api

                                            oracle_centerlines_list.append(interpolated_centerline)

                                    except:
                                        # TODO: Take the closest max_points if the previous algorithm fails
                                        
                                        map_features_utils_instance.debug_centerline_and_agent([relevant_centerline_orig], agent_xy, obs_len, obs_len+pred_len, split_name, file_id)
                                        pdb.set_trace()
                                        wrong_centerlines.append(file_id)            
                
                                else:
                                    # print("The algorithm has the exact number of points")
                                    relevant_centerlines_filtered.append(relevant_centerline_filtered)            

                                    # Store oracle

                                    if index_centerline == 0 and algorithm == "map_api": 
                                        # The first centerline also corresponds to the oracle using map_api

                                        oracle_centerlines_list.append(relevant_centerline_filtered)

                            seq_map_info["relevant_centerlines_filtered"] = relevant_centerlines_filtered

                            fig, ax = plt.subplots(figsize=(6,6), facecolor="black")

                            xmin = ymin = 50000
                            xmax = ymax = -50000

                            # Paint centerlines

                            for centerline_coords in relevant_centerlines_filtered:
                                # visualize_centerline(centerline_coords) # Uncomment this to check the start and end

                                lane_polygon = centerline_to_polygon(centerline_coords)

                                if np.min(lane_polygon[:,0]) < xmin: xmin = np.min(lane_polygon[:,0])
                                if np.min(lane_polygon[:,1]) < ymin: ymin = np.min(lane_polygon[:,1])
                                if np.max(lane_polygon[:,0]) > xmax: xmax = np.max(lane_polygon[:,0])
                                if np.max(lane_polygon[:,1]) > ymax: ymax = np.max(lane_polygon[:,1])
                                                                        #"black"  
                                ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "white", edgecolor='white', fill=True)
                                ax.plot(centerline_coords[:, 0], centerline_coords[:, 1], "-", color="gray", linewidth=1.5, alpha=1.0, zorder=2)

                            center_plausible_area_filtered = np.zeros((2))
                            center_plausible_area_filtered[0] = (xmin+xmax)/2
                            center_plausible_area_filtered[1] = (ymin+ymax)/2

                            real_world_width = xmax - xmin
                            real_world_height = ymax - ymin

                            seq_map_info["center_plausible_area_filtered"] = center_plausible_area_filtered
                            seq_map_info["real_world_width"] = real_world_width
                            seq_map_info["real_world_height"] = real_world_height

                            # circ_car = plt.Circle((center_plausible_area_filtered[0], center_plausible_area_filtered[1]), 0.2, color="cyan", fill=True)
                            # ax.add_patch(circ_car)

                            # Paint agent's orientation

                            dx = lane_dir_vector[0] * 4
                            dy = lane_dir_vector[1] * 4

                            plt.xlim(xmin, xmax)
                            plt.ylim(ymin, ymax)
                            plt.axis("off")

                            filename = os.path.join(output_dir,f"{file_id}_binary_plausible_area_filtered_gray.png")
                            # plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
                            
                            if SAVE_BEV_PLAUSIBLE_AREA: 
                                fig.tight_layout(pad=0)
                                fig.canvas.draw()
                                img_gray = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2GRAY)
                                cv2.imwrite(filename, img_gray)
                            # plt.close('all')

                            plt.arrow(
                                agent_xy[obs_len-1,0], # Arrow origin
                                agent_xy[obs_len-1,1],
                                dx, # Length of the arrow
                                dy,
                                color="darkmagenta",
                                width=0.3,
                                zorder=16,
                            )

                            if agent_xy.shape[0] == obs_len+pred_len: # train and val
                                # Agent observation

                                plt.plot(
                                    agent_xy[:obs_len, 0],
                                    agent_xy[:obs_len, 1],
                                    "-",
                                    color="b",
                                    alpha=1,
                                    linewidth=3,
                                    zorder=15,
                                )

                                # Agent prediction

                                plt.plot(
                                    agent_xy[obs_len:, 0],
                                    agent_xy[obs_len:, 1],
                                    "-",
                                    color="r",
                                    alpha=1,
                                    linewidth=3,
                                    zorder=15,
                                )

                                final_x = agent_xy[-1, 0]
                                final_y = agent_xy[-1, 1]

                                # Final position

                                plt.plot(
                                    final_x,
                                    final_y,
                                    "o",
                                    color="r",
                                    alpha=1,
                                    markersize=5,
                                    zorder=15,
                                )
                            else: # test
                                # Agent observation
                                
                                plt.plot(
                                    agent_xy[:, 0],
                                    agent_xy[:, 1],
                                    "-",
                                    color="b",
                                    alpha=1,
                                    linewidth=3,
                                    zorder=15,
                                )

                                final_x = agent_xy[-1, 0]
                                final_y = agent_xy[-1, 1]

                                # Final position

                                plt.plot(
                                    final_x,
                                    final_y,
                                    "o",
                                    color="b",
                                    alpha=1,
                                    markersize=5,
                                    zorder=15,
                                )

                            filename = os.path.join(output_dir,f"{file_id}_binary_plausible_area_filtered_color.png")
                            # plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

                            lane_polygon = centerline_to_polygon(relevant_centerlines_filtered[0])
                            ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "green", edgecolor='green', fill=True)
                            
                            if SAVE_BEV_PLAUSIBLE_AREA: 
                                fig.tight_layout(pad=0)
                                fig.canvas.draw()
                                img_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                                cv2.imwrite(filename, img_bgr)
                            plt.close('all')

                            # plt.xlabel("Map X")
                            # plt.ylabel("Map Y")
                            # plt.axis("on")
                            # plt.title(f"Number of candidates = {len(relevant_centerlines_filtered)}")   
                            # plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)

                            end_ = time.time()

                            # print("Time consumed: ", end_-start_)

                            map_info[str(file_id)] = seq_map_info 

                        elif mode == "train": # only best centerline ("oracle")
                            start_ = time.time()

                            oracle_centerline = map_feature_helpers["ORACLE_CENTERLINE"]                     

                            # Get index of the closest waypoint to the first observation
                            
                            if first_centerline_waypoint == "last_obs":
                                closest_wp_first, _ = map_features_utils_instance.get_closest_wp(last_obs, relevant_centerline)
                            elif first_centerline_waypoint == "first_obs":
                                closest_wp_first, _ = map_features_utils_instance.get_closest_wp(first_obs, relevant_centerline)
                            
                            # Get index of the closest waypoint to the last observation + dist_around
                            
                            closest_wp_last, dist_array_last = map_features_utils_instance.get_closest_wp(last_obs, oracle_centerline)
                            dist_array_last = dist_array_last[closest_wp_last:] # To determine dist around from this point

                            idx, value = find_nearest(dist_array_last,dist_around)
                            num_points = idx + (closest_wp_last - closest_wp_first)
                            if num_points < min_points:
                                    num_points = min_points # you must have at least 4 points to conduct a cubic interpolation

                            # Determine if the best oracle is inverted (this sometimes happens if the corresponding agent is
                            # carrying a lane change maneuver): The best and most interpretable solution is if the
                            # the end point is closer than the start point and the last observation is closer to the start
                            # rather than the first observation

                            oracle_centerline_orig = copy.deepcopy(oracle_centerline)
                            
                            ec = euclidean_distances(oracle_centerline[:-1,:],oracle_centerline[1:,:])
                            indeces_rows = indeces_cols = np.arange(ec.shape[0])
                            dist = ec[indeces_rows,indeces_cols]

                            invert = False

                            try:
                                dist_firstobs2start = np.cumsum(dist[:closest_wp_first+1])[-1]
                                dist_lastobs2start = np.cumsum(dist[:closest_wp_last+1])[-1]
                                dist_firstobs2end = np.cumsum(dist[closest_wp_first:])[-1]

                                if ((dist_lastobs2start >= dist_firstobs2start)
                                    and ((closest_wp_first > 0 and closest_wp_first < oracle_centerline.shape[0])
                                        or (dist_firstobs2start <= dist_firstobs2end))):
                                    pass
                                else:
                                    invert = True

                            except Exception as e:
                                # print("e: ", e)
                                invert = True

                            if invert:
                                # print("invert oracle")

                                oracle_centerline_orig = copy.deepcopy(oracle_centerline)
                                oracle_centerline = oracle_centerline[::-1]

                                # Repeat again the process to obtain the closest wps

                                # Get index of the closest waypoint to the first observation
                            
                                closest_wp_first, _ = map_features_utils_instance.get_closest_wp(first_obs, oracle_centerline)

                                # Get index of the closest waypoint to the last observation + dist_around
                            
                                closest_wp_last, dist_array_last = map_features_utils_instance.get_closest_wp(last_obs, oracle_centerline)
                                dist_array_last = dist_array_last[closest_wp_last:]

                                idx, value = find_nearest(dist_array_last,dist_around) # TODO: Take the next point, or an interpolated point
                                # between the closest and the next

                                num_points = idx + (closest_wp_last - closest_wp_first)
                                if num_points < min_points:
                                    num_points = min_points # you must have at least 4 points to conduct a cubic interpolation

                            # Reduce the lane to the closest N points, starting from the first observation 
                            # (closest to current position) and ending in the closest wp assuming CTRA during pred seconds

                            end_point = agent_xy[-1,:]

                            if closest_wp_first+num_points+1 <= oracle_centerline.shape[0]:
                                oracle_centerline_filtered = oracle_centerline[closest_wp_first:closest_wp_first+num_points+1,:]
                            else: # If we have reached the end, travel backwards 
                                    # TODO: Interpolate frontwards
                                back_num_points = (closest_wp_first+num_points+1) - oracle_centerline.shape[0]
                                print("back: ", back_num_points)
                                oracle_centerline_filtered = oracle_centerline[closest_wp_first-back_num_points:,:]
                            
                            if oracle_centerline_filtered.shape[0] != max_points:
                                try:
                                    # start = time.time()
                                    interpolated_centerline = map_features_utils_instance.interpolate_centerline(relevant_centerline_filtered,
                                                                                                                    max_points=max_points,
                                                                                                                    agent_xy=agent_xy,
                                                                                                                    obs_len=obs_len,
                                                                                                                    seq_len=obs_len+pred_len,
                                                                                                                    split=split_name,
                                                                                                                    seq_id=file_id,
                                                                                                                    viz=viz)
                                    # end = time.time()
                                    # print("Time consumed by interpolation: ", end-start)

                                    assert interpolated_centerline.shape[0] == max_points

                                    oracle_centerlines_list.append(interpolated_centerline)
                                except:
                                    # TODO: Take the closest max_points if the previous algorithm fails
                                    
                                    map_features_utils_instance.debug_centerline_and_agent([oracle_centerline_orig], agent_xy, obs_len, obs_len+pred_len, split_name, file_id)
                                    pdb.set_trace()
                                    wrong_centerlines.append(file_id)            
                
                            else:
                                # print("The algorithm has the exact number of points")
                                oracle_centerlines_list.append(oracle_centerline_filtered)
                                
                            end_ = time.time()
                            # print("Time consumed by L2 norm: ", end_-start_)

                    end = time.time()

                    aux_time += (end-start)
                    time_per_iteration = aux_time/(i+1)
                    
                    if i % check_every_n_files == 0:
                        print(f"Time per iteration: {time_per_iteration} s. \n \
                                Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")
                        print("Wrong centerlines: ", wrong_centerlines) 

            # TODO: At this moment, both train and test use the same algorithm

            # Save only the oracle (best possible centerline) as a np.array -> num_sequences x max_points x 2 
            if mode == "train":
                filename = os.path.join(BASE_DIR,config.dataset.path,split_name,
                                        f"data_processed_{str(int(features[1]*100))}_percent","oracle_centerlines.npy")
                with open(filename, 'wb') as my_file: np.save(my_file, oracle_centerlines_list)

            # Save N centerlines per sequence. Note that the number of variables per sequence may vary
            elif mode == "test":
                filename = os.path.join(BASE_DIR,config.dataset.path,split_name,
                                        f"data_processed_{str(int(features[1]*100))}_percent",
                                        f"relevant_centerlines_{first_centerline_waypoint}_{str(max_points)}_points.npz")
                with open(filename, 'wb') as my_file: np.savez(my_file, map_info)

                # If method is least_squares and map_api, we assume the first centerline returned by the 
                # algorithm will be also the oracle (most plausible)

                if algorithm == "map_api":
                    filename = os.path.join(BASE_DIR,config.dataset.path,split_name,
                                        f"data_processed_{str(int(features[1]*100))}_percent",
                                        f"oracle_centerlines_{first_centerline_waypoint}_{str(max_points)}_points.npy")
                    oracle_centerlines_array = np.array(oracle_centerlines_list)
                    with open(filename, 'wb') as my_file: np.save(my_file, oracle_centerlines_array)

            
            