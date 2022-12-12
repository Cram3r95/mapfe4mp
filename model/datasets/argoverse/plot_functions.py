#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Visualization functions

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo
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
        output_predictions_abs: np.ndarray, # Around 0,0
        output_confidences: np.ndarray,
        output_agent_orientation: float, 
        gt_abs: np.ndarray, # Around 0,0
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
        plot_agent_yaw: bool = False,
        plot_output_confidences: bool = False,
        worst_scenes: list = [],
        check_data_aug: bool = False
) -> None:

    
    splits_name = ["train","val","test"]
    split_name = results_path.split('/')[-1]
    assert split_name in splits_name, "Wrong results_path!"

    # https://www.computerhope.com/htmcolor.htm#color-codes
    color_dict_obs = {"AGENT": "#ECA154", "OTHER": "#413839", "AV": "#0000A5"}
    color_dict_pred_gt = {"AGENT": "#d33e4c", "OTHER": "#686A6C", "AV": "#157DEC"}

    num_agents = input_abs.shape[0]
    num_modes = output_predictions_abs.shape[0]
    obs_len = input_abs.shape[1]
    pred_len = gt_abs.shape[1]

    # Transform to global coordinates

    input = input_abs + map_origin
    if np.any(output_predictions_abs):
        output_predictions = output_predictions_abs + map_origin
    gt = gt_abs + map_origin

    _, num_centerlines, points_per_centerline, data_dim = relevant_centerlines_abs.shape
    rows,cols,_ = np.where(relevant_centerlines_abs[:,:,:,0] == 0.0) # only sum the origin to non-padded centerlines
    relevant_centerlines = relevant_centerlines_abs + map_origin
    relevant_centerlines[rows,cols,:,:] = np.zeros((points_per_centerline,data_dim))

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
                  fontdict=font, backgroundcolor='silver')
    # else:
    #     plt.axis("off")

    # Social information
    
    for i in range(num_agents): # Sequences (.csv)
        object_type = translate_object_type(int(object_class_list[i]))

        # Observation

        plt.plot(
            input[i, :, 0],
            input[i, :, 1],
            color=color_dict_obs[object_type],
            label="Input",
            alpha=1,
            linewidth=3,
            zorder=15,
        )

        # Last observation

        plt.plot(
            input[i, -1, 0],
            input[i, -1, 1],
            "o",
            color=color_dict_obs[object_type],
            label="Input",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        if plot_agent_yaw:
            ori = str(np.around(output_agent_orientation,2))
            if object_type == "AGENT":
                plt.text(
                    input[i, -1, 0] + 1,
                    input[i, -1, 1] + 1,
                    f"yaw = {ori}",
                    fontsize=12,
                    zorder=20
                    )

        # Groundtruth prediction

        if split_name != "test":
            plt.plot(
                gt[i, :, 0],
                gt[i, :, 1],
                color=color_dict_pred_gt[object_type],
                label="GT",
                alpha=1,
                linewidth=3,
                zorder=20,
            )

            # Groundtruth end-point

            plt.plot(
                gt[i, -1, 0],
                gt[i, -1, 1],
                "D",
                color=color_dict_pred_gt[object_type],
                label="GT",
                alpha=1,
                linewidth=3,
                zorder=20,
                markersize=9,
            )

        # Model predictions

        if object_type == "AGENT" and num_modes > 0:
            # Multimodal prediction (only AGENT of interest)
            
            sorted_confidences = np.sort(output_confidences)
            slope = (1 - 1/num_modes) / (num_modes - 1)
            
            for num_mode in range(num_modes):
                
                _, conf_index = np.where(output_confidences[0,num_mode] == sorted_confidences)
                transparency = round(slope * (conf_index.item()+1),2)
                
                # Prediction
                plt.plot(
                    output_predictions[num_mode, :, 0],
                    output_predictions[num_mode, :, 1],
                    color="#007672",
                    label="Output",
                    alpha=transparency,
                    linewidth=3,
                    zorder=15,
                )
                # Prediction endpoint
                plt.plot(
                    output_predictions[num_mode, -1, 0],
                    output_predictions[num_mode, -1, 1],
                    "*",
                    color="#007672",
                    label="Output",
                    alpha=transparency,
                    linewidth=3,
                    zorder=15,
                    markersize=9,
                )
                
                if plot_output_confidences:
                    # Confidence
                    plt.text(
                            output_predictions[num_mode, -1, 0] + 1,
                            output_predictions[num_mode, -1, 1] + 1,
                            str(round(output_confidences[0,num_mode],2)),
                            zorder=21,
                            fontsize=9
                            )

    # Relevant centerlines
    
    count_rep_centerlines = np.zeros((num_centerlines))
    rep_centerlines = []

    # TODO: Prepare this code to deal with batch size != 1
    if len(relevant_centerlines) > 0:
        for id_centerline in range(num_centerlines):
            centerline = relevant_centerlines[0,id_centerline,:,:] # TODO: We assume batch_size = 1 here

            if not np.any(centerline): # Avoid plotting padded centerlines
                continue
            
            # Check repeated centerlines

            flag_repeated = False
            for index_centerline, aux_centerline in enumerate(rep_centerlines):
                if np.allclose(centerline,aux_centerline):
                    flag_repeated = True
                    count_rep_centerlines[index_centerline] += 1
            if not flag_repeated:
                count_rep_centerlines[id_centerline] += 1
                rep_centerlines.append(centerline)

            # Centerline

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

        # Plot the number of repetitions for each centerline

        for index_repeated in range(len(count_rep_centerlines)):
            if count_rep_centerlines[index_repeated] > 0:
                centerline = relevant_centerlines[0,index_repeated,:,:] # TODO: We assume batch_size = 1 here

                plt.text(
                        centerline[-1, 0] + 1,
                        centerline[-1, 1] + 1,
                        str(int(count_rep_centerlines[index_repeated])),
                        fontsize=12
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

        if check_data_aug:
            filename = os.path.join(results_path,subfolder,str(seq_id)+"_data_aug.png")
        else:
            filename = os.path.join(results_path,subfolder,str(seq_id)+".png")
            
        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

    plt.cla()
    plt.close('all')