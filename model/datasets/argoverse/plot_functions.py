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

# Main plot function

def plot_trajectories_custom(filename,results_path,obs_traj,map_origin,object_class_id_list,dist_rasterized_map,
                      rot_angle=-1,obs_len=20,smoothen=False,save=False,data_aug=False,
                      pred_trajectories=torch.zeros((0)),ade_metric=None,fde_metric=None,change_bg=False):
    """
    N.B. All inputs must be torch tensors in CPU, not GPU (in order to plot the trajectories)!

    It is assumed batch_size = 1, that is, all variables must represent a single sequence!

    Plot until obs_len points per trajectory. If obs_len != None, we
    must distinguish between observation (color with a marker) and prediction (same color with another marker)
    """

    mode = "k1" # unimodal (by default)

    seq_len = obs_traj.shape[0]
    if seq_len == obs_len:
        plot_len = 1
    elif seq_len > obs_len:
        plot_len = 2

    if len(pred_trajectories.shape) > 1: # != torch.zeros((0))
        if pred_trajectories.shape[1] == 1: # Output from our model (pred_len x 1 x 2 -> Unimodal)
            plot_len = 3
        else: # Output from our model (pred_len x N x 2 -> Multimodal)
            plot_len = plot_len + pred_trajectories.shape[1] 
            mode = f"k{pred_trajectories.shape[1]}"

    xcenter, ycenter = map_origin[0].item(), map_origin[1].item()

    x_min = xcenter + dist_rasterized_map[0]
    x_max = xcenter + dist_rasterized_map[1]
    y_min = ycenter + dist_rasterized_map[2]
    y_max = ycenter + dist_rasterized_map[3]

    plot_object_trajectories = True
    plot_object_heads = True

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white", frameon=False)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Comment if you want to generate images with axis
    if ade_metric and fde_metric:
        font = {
                'family': 'serif',
                'color':  'blue',
                'weight': 'normal',
                'size': 16,
               }
        plt.title(f"minADE: {round(ade_metric,3)} ; minFDE: {round(fde_metric,3)}", \
                  fontdict=font, backgroundcolor= 'silver')
    t0 = time.time()

    object_type_tracker: Dict[int, int] = defaultdict(int)

    seq_list = []

    for i in range(len(object_class_id_list)):
        abs_traj_ = obs_traj[:,i,:].view(-1,2).unsqueeze(dim=1) # obs_len/seq_len x 1 (emulating batch) x 2 (rel-rel)
        obj_id = object_class_id_list[i]
        seq_list.append([obj_id,abs_traj_])

        if obj_id == 1: # Agent
            if seq_len == obs_len:
                obs_ = abs_traj_
            elif seq_len > obs_len:
                obs_ = abs_traj_[:obs_len,:,:]

            if len(pred_trajectories.shape) > 1: # != torch.zeros((0))
                if pred_trajectories.shape[1] == 1: # Output from our model (pred_len x 1 x 2 -> Unimodal
                    agent_traj_with_pred = torch.cat((obs_,pred_trajectories),dim=0)
                    agent_pred_abs = agent_traj_with_pred[obs_len:,:]
                else: # Output from our model (pred_len x N x 2 -> Multimodal)
                    agent_pred_abs = []

                    for t in range(pred_trajectories.shape[1]): # num_modes
                        pred_traj_ = pred_trajectories[:,t,:].unsqueeze(dim=1) # 30 x 1 x 2 (each mode)
                        agent_traj_with_pred = torch.cat((obs_,pred_traj_),dim=0)
                        agent_pred_abs_ = agent_traj_with_pred[obs_len:,:]

                        agent_pred_abs.append(agent_pred_abs_)

    # Plot all the tracks up till current frame

    for seq_id in seq_list:
        object_type = translate_object_type(int(seq_id[0]))
        seq_abs = seq_id[1]

        if seq_len > obs_len:
            obs_x, obs_y = seq_abs[:obs_len,0,0] + xcenter, seq_abs[:obs_len,0,1] + ycenter
            pred_gt_x, pred_gt_y = seq_abs[obs_len:,0,0] + xcenter, seq_abs[obs_len:,0,1] + ycenter
        elif seq_len == obs_len:
            obs_x, obs_y = seq_abs[:,0,0] + xcenter, seq_abs[:,0,1] + ycenter

        i_mm = 0

        if plot_object_trajectories:
            for i in range(plot_len):
                if i == 0: # obs
                    marker_type_index = 0
                    cor_x, cor_y = obs_x, obs_y 
                elif i == 1 and (seq_len > obs_len): # pred (gt)
                    marker_type_index = 1
                    cor_x, cor_y = pred_gt_x, pred_gt_y
                elif i >= 2 and object_type == "AGENT": # pred (model) At this moment, only target AGENT
                    marker_type_index = 2
                    if pred_trajectories.shape[1] == 1: # Unimodal
                        cor_x, cor_y = agent_pred_abs[:,0,0] + xcenter, agent_pred_abs[:,0,1] + ycenter
                    else: # Multimodal
                        agent_pred_abs_ = agent_pred_abs[i_mm]
                        cor_x, cor_y = agent_pred_abs_[:,0,0] + xcenter, agent_pred_abs_[:,0,1] + ycenter
                        i_mm += 1
                else:
                    continue

                colour,linewidth,marker_type,marker_size,transparency = MARKER_DICT[object_type][marker_type_index]

                if smoothen:
                    if cor_x.device.type != "cpu": cor_x = cor_x.to("cpu")
                    if cor_y.device.type != "cpu": cor_y = cor_y.to("cpu")

                    polyline = np.column_stack((cor_x, cor_y))
                    num_points = cor_x.shape[0] * 3
                    smooth_polyline = interpolate_polyline(polyline, num_points)

                    cor_x, cor_y = smooth_polyline[:,0], smooth_polyline[:,1]

                    plt.plot(
                        cor_x,
                        cor_y,
                        "-",
                        color=colour,
                        label=object_type if not object_type_tracker[object_type] else "",
                        alpha=transparency,
                        linewidth=linewidth,
                        zorder=ZORDER[object_type],
                    )
                else:
                    if cor_x.device.type != "cpu": cor_x = cor_x.to("cpu")
                    if cor_y.device.type != "cpu": cor_y = cor_y.to("cpu")
                    plt.scatter(cor_x, cor_y, c=colour, s=scatter_size, alpha=transparency)

                if plot_object_heads:
                    final_pos_x, final_pos_y = cor_x[-1], cor_y[-1]
                    
                    plt.plot(
                        final_pos_x,
                        final_pos_y,
                        marker_type,
                        color=colour,
                        label=object_type if not object_type_tracker[object_type] else "",
                        alpha=transparency,
                        markersize=marker_size,
                        zorder=ZORDER[object_type],
                    )

        object_type_tracker[object_type] += 1

    # Merge local driveable information and trajectories information
    
    ## Read map

    img_map = cv2.imread(filename)
    if rot_angle == 90:
        img_map = cv2.rotate(img_map, cv2.ROTATE_90_CLOCKWISE)
    elif rot_angle == 180:
        img_map = cv2.rotate(img_map, cv2.ROTATE_180)
    elif rot_angle == 270 or rot_angle == -90:
        img_map = cv2.rotate(img_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: # -1 (default)
        rot_angle = 0

    ## Foreground (Trajectories)

    ### TODO: Is it possible to optimize this saving without padding? save -> read -> destroy

    plt.savefig("evaluate/argoverse/aux.png", bbox_inches='tight', facecolor=fig.get_facecolor(), 
                edgecolor='none', pad_inches=0)
    img_trajectories = cv2.imread("evaluate/argoverse/aux.png")
    os.system("rm -rf evaluate/argoverse/aux.png")
    img_trajectories = cv2.resize(img_trajectories, img_map.shape[:2][::-1]) # ::-1 to invert (we need width,height, not height,width!) 
    img2gray = cv2.cvtColor(img_trajectories,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,253,255,cv2.THRESH_BINARY_INV)
    img2_fg = cv2.bitwise_and(img_trajectories,img_trajectories,mask=mask)

    ## Background (Map lanes)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_map,img_map,mask=mask_inv)

    ## Merge
 
    if abs(dist_rasterized_map[0]) == 40: # Right now the preprocessed map images are generated
        # based on a [40,-40,40,-40] top view image. Then, we can merge the information
        full_img_cv = cv2.add(img1_bg,img2_fg)
    else: # TODO: Preprocess here the map? That would be too slow ...
        full_img_cv = img_trajectories 

    if save:
        curr_seq = filename.split('/')[-1].split('.')[0]

        output_dir = os.path.join(results_path,"data_images_trajs/")
        # results_path = os.path.join(*filename.split('/')[:-2],"data_images_trajs/")
        if not os.path.exists(output_dir):
            print("Create trajs folder: ", output_dir)
            os.makedirs(output_dir) # makedirs creates intermediate folders
 
        if data_aug:
            filename2 = output_dir + curr_seq + "_" + str(rot_angle) + "_aug" + ".png"
        elif len(agent_traj_with_pred) != 1:
            filename2 = output_dir + curr_seq + f"_qualitative_{mode}" + ".png"
        else:
            filename2 = output_dir + curr_seq + "_original" + ".png"

        cv2.imwrite(filename2,full_img_cv)

        if change_bg:
            full_img_cv_inverted = change_bg_color(full_img_cv)

            aux = filename2.split('.png')[0]
            filename2_inv = aux+"_inverted.png"
            cv2.imwrite(filename2_inv,full_img_cv_inverted)

    resized_full_img_cv = cv2.resize(full_img_cv,(224,224))
    norm_resized_full_img_cv = resized_full_img_cv / 255.0
    
    plt.cla()
    plt.close('all')

    return norm_resized_full_img_cv

## Plot standard Argoverse

def viz_predictions(
        seq_id: int,
        results_path: str,
        input_abs: np.ndarray, # Around 0,0
        output_abs: np.ndarray, # Around 0,0
        target_abs: np.ndarray, # Around 0,0
        city_names: np.ndarray,
        map_origin,
        avm,
        centerlines: np.ndarray = np.array([]),
        show: bool = False,
        save: bool = False,
        ade_metric: float = None,
        fde_metric: float = None,
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

    query_search_range_manhattan_ = 5

    num_tracks = input_abs.shape[0]
    obs_len = input_abs.shape[1]
    pred_len = target_abs.shape[1]

    # Transform to global coordinates

    input_ = input_abs + map_origin
    output = output_abs + map_origin
    target = target_abs + map_origin

    fig = plt.figure(0, figsize=(8, 7))

    if ade_metric and fde_metric:
        font = {
                'family': 'serif',
                'color':  'blue',
                'weight': 'normal',
                'size': 16,
               }
        plt.title(f"minADE: {round(ade_metric,3)} ; minFDE: {round(fde_metric,3)}", \
                  fontdict=font, backgroundcolor= 'silver')

    for i in range(num_tracks): # Sequences (.csv)
        # Observation
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
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
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        # Groundtruth prediction
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
        )
        # Groundtruth end-point
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
            markersize=9,
        )
        # Centerlines (Optional)
        if len(centerlines) > 0:
            for j in range(len(centerlines[i])):
                plt.plot(
                    centerlines[i][j][:, 0],
                    centerlines[i][j][:, 1],
                    "--",
                    color="grey",
                    alpha=1,
                    linewidth=1,
                    zorder=0,
                )
        # Multimodal prediction
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
                "o",
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            for k in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    output[num_mode, k, 0],
                    output[num_mode, k, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )
        # Identify lanes using input data
        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                input_[i, j, 0],
                input_[i, j, 1],
                city_names[i],
                query_search_range_manhattan=query_search_range_manhattan_,
            )
            # Paint lanes
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        # Identify lanes using prediction data
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                target[i, j, 0],
                target[i, j, 1],
                city_names[i],
                query_search_range_manhattan=query_search_range_manhattan_,
            )
            # Paint lanes
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        if show:
            plt.show()

        if save:
            filename = os.path.join(results_path,"data_images_trajs",str(seq_id)+".png")
            plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

        plt.cla()
        plt.close('all')

def viz_predictions_all(
        seq_id: int,
        results_path: str,
        input_abs: np.ndarray, # Around 0,0
        output_abs: np.ndarray, # Around 0,0
        target_abs: np.ndarray, # Around 0,0
        object_class_list: np.ndarray,
        city_name: str,
        map_origin,
        avm,
        dist_rasterized_map: int = None,
        centerlines: np.ndarray = np.array([]),
        show: bool = False,
        save: bool = False,
        ade_metric: float = None,
        fde_metric: float = None,
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

    plt.axis("off")

    if ade_metric and fde_metric:
        font = {
                'family': 'serif',
                'color':  'blue',
                'weight': 'normal',
                'size': 16,
               }
        plt.title(f"minADE: {round(ade_metric,3)} ; minFDE: {round(fde_metric,3)}", \
                  fontdict=font, backgroundcolor= 'silver')

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
        
        if len(centerlines) > 0:
            for j in range(len(centerlines[i])):
                plt.plot(
                    centerlines[i][j][:, 0],
                    centerlines[i][j][:, 1],
                    "--",
                    color="grey",
                    alpha=1,
                    linewidth=1,
                    zorder=0,
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
        output_dir = os.path.join(results_path,"data_images_trajs/")
        if not os.path.exists(output_dir):
            print("Create trajs folder: ", output_dir)
            os.makedirs(output_dir) # makedirs creates intermediate folders

        filename = os.path.join(results_path,"data_images_trajs",str(seq_id)+".png")
        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

    plt.cla()
    plt.close('all')