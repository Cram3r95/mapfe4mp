#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Data augmentation functions

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import pdb
import os
import time
from collections import defaultdict, OrderedDict
from typing import Dict

# DL & Math imports

import numpy as np
import torch
import cv2
import scipy.interpolate as interp

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

import model.datasets.argoverse.dataset_utils as dataset_utils
import model.datasets.argoverse.goal_points_functions as goal_points_functions

#######################################

# https://matplotlib.org/stable/api/markers_api.html
# https://matplotlib.org/stable/gallery/color/named_colors.html

COLORS = "darkviolet"
COLORS_MM = ["darkviolet", "orange", "chartreuse"]

                        # Observation / Prediction (GT) / Prediction (Model)                   
                        # (color, linewidth, type, size, transparency)
MARKER_DICT = {"AGENT":  (("b",6,"o",11,1.0),("c",6,"*",11,0.5),("darkviolet",6,"*",11,1.0)),
               "OTHER":  (("g",4,"o",10,1.0),("g",4,"*",10,0.5)),
               "AV":     (("r",4,"o",10,1.0),("m",4,"*",10,1.0))}
scatter_size = 2  

ZORDER = {"AGENT": 3, "OTHER": 3, "AV": 3}

# Aux functions

def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """
    """
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def translate_object_type(int_id):
    """
    """
    if int_id == 0:
        return "AV"
    elif int_id == 1:
        return "AGENT"
    else:
        return "OTHER"

def change_bg_color(img):
    """
    """
    img[np.all(img == (0, 0, 0), axis=-1)] = (255,255,255)

    return img

# Main plot function (Standard visualization in Argoverse)

def viz_predictions_all(
        seq_id: int,
        results_path: str,
        input_abs: np.ndarray, # Around 0,0
        output_abs: np.ndarray, # Around 0,0
        target_abs: np.ndarray, # Around 0,0
        object_class_list: np.ndarray,
        city_name: str,
        map_origin: np.ndarray,
        avm,
        dist_rasterized_map: int = None,
        relevant_centerlines_abs: np.ndarray = np.array([]),
        show: bool = False,
        save: bool = False,
        ade_metric: float = None,
        fde_metric: float = None,
        worst_scenes: list = [],
) -> None:
    """Visualize predicted trjectories.
    Args:
        OBS: Track = Sequence

        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show
    """
    
    splits_name = ["train","val","test"]
    split_name = results_path.split('/')[-1]
    assert split_name in splits_name, "Wrong results_path!"

    # https://www.computerhope.com/htmcolor.htm#color-codes
    color_dict_obs = {"AGENT": "#ECA154", "OTHER": "#413839", "AV": "#0000A5"}
    color_dict_pred_gt = {"AGENT": "#d33e4c", "OTHER": "#686A6C", "AV": "#157DEC"}

    query_search_range_manhattan_ = 2.5

    num_agents = input_abs.shape[0]
    obs_len = input_abs.shape[1]
    pred_len = target_abs.shape[1]

    # Transform to global coordinates

    input_ = input_abs + map_origin
    output = output_abs + map_origin
    target = target_abs + map_origin

    fig = plt.figure(0, figsize=(8,8))

    x_min = map_origin[0] - dist_rasterized_map
    x_max = map_origin[0] + dist_rasterized_map
    y_min = map_origin[1] - dist_rasterized_map
    y_max = map_origin[1] + dist_rasterized_map

    if ade_metric and fde_metric: # Debug 
        font = {
                'family': 'serif',
                'color':  'blue',
                'weight': 'normal',
                'size': 16,
               }
        plt.title(f"minADE: {round(ade_metric,3)} ; minFDE: {round(fde_metric,3)}", \
                  fontdict=font, backgroundcolor= 'silver')
    else:
        plt.axis("off")

    for i in range(num_agents): # Sequences (.csv)
        object_type = translate_object_type(int(object_class_list[i]))

        # Observation

        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color=color_dict_obs[object_type],
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )

        # Last observation

        plt.plot(
            input_[i, -1, 0],
            input_[i, -1, 1],
            "o",
            color=color_dict_obs[object_type],
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )

        # Groundtruth prediction

        if split_name != "test":
            plt.plot(
                target[i, :, 0],
                target[i, :, 1],
                color=color_dict_pred_gt[object_type],
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=20,
            )

            # Groundtruth end-point

            plt.plot(
                target[i, -1, 0],
                target[i, -1, 1],
                "D",
                color=color_dict_pred_gt[object_type],
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=20,
                markersize=9,
            )

        # Centerlines (Optional)
        
        # TODO: Prepare this code to deal with batch size != 1
        if len(relevant_centerlines_abs) > 0:

            for j in range(len(relevant_centerlines_abs[:,0,:,:])):
                centerline = relevant_centerlines_abs[j,0,:,:]
                plt.plot(
                    centerline[:, 0],
                    centerline[:, 1],
                    "--",
                    color="black",
                    alpha=1,
                    linewidth=1,
                    zorder=16,
                )

                # Goal point (end point of plausible centerline)

                plt.plot(
                    centerline[-1, 0],
                    centerline[-1, 1],
                    "*",
                    color="black",
                    alpha=1,
                    linewidth=1,
                    zorder=16,
                )

        if object_type == "AGENT":
            # Multimodal prediction (only AGENT of interest)
            for num_mode in range(output.shape[0]):
                # Prediction
                plt.plot(
                    output[num_mode, :, 0],
                    output[num_mode, :, 1],
                    color="#007672",
                    label="Predicted",
                    alpha=1,
                    linewidth=3,
                    zorder=15,
                )
                # Prediction endpoint
                plt.plot(
                    output[num_mode, -1, 0],
                    output[num_mode, -1, 1],
                    "*",
                    color="#007672",
                    label="Predicted",
                    alpha=1,
                    linewidth=3,
                    zorder=15,
                    markersize=9,
                )

    seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    ### Get lane centerlines which lie within the range of trajectories

    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min):

            avm.draw_lane(lane_id, city_name)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.axis("equal")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    if show:
        plt.show()

    if save:
        if worst_scenes:
            subfolder = "data_images_worst_trajs"
        else:
            subfolder = "data_images_trajs"

        output_dir = os.path.join(results_path,subfolder)
        if not os.path.exists(output_dir):
            print("Create trajs folder: ", output_dir)
            os.makedirs(output_dir) # makedirs creates intermediate folders

        filename = os.path.join(results_path,subfolder,str(seq_id)+".png")
        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

    plt.cla()
    plt.close('all')