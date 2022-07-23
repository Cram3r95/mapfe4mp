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
from collections import defaultdict
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
scatter_size = 10  

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

def plot_trajectories(filename,traj_rel,first_obs,map_origin,object_class_id_list,dist_rasterized_map,
                      rot_angle=-1,obs_len=20,smoothen=False,save=False,data_aug=False,
                      pred_trajectories_rel=torch.zeros((0)),ade_metric=None,fde_metric=None):
    """
    N.B. All inputs must be torch tensors in CPU, not GPU (in order to plot the trajectories)!

    It is assumed batch_size = 1, that is, all variables must represent a single sequence!

    Plot until obs_len points per trajectory. If obs_len != None, we
    must distinguish between observation (color with a marker) and prediction (same color with another marker)
    """

    mode = "k1" # unimodal

    seq_len = traj_rel.shape[0]
    if seq_len == obs_len:
        plot_len = 1
    elif seq_len > obs_len:
        plot_len = 2

    if len(pred_trajectories_rel.shape) > 1: # != torch.zeros((0))
        if pred_trajectories_rel.shape[1] == 1: # Output from our model (pred_len x 1 x 2 -> Unimodal)
            plot_len = 3
        else: # Output from our model (pred_len x N x 2 -> Multimodal)
            plot_len = plot_len + pred_trajectories_rel.shape[1] 
            mode = f"k{pred_trajectories_rel.shape[1]}"

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
        plt.title(f"ADE: {round(ade_metric,3)} ; FDE: {round(fde_metric,3)}", \
                  fontdict=font, backgroundcolor= 'silver')
    t0 = time.time()

    object_type_tracker: Dict[int, int] = defaultdict(int)

    seq_list = []

    for i in range(len(object_class_id_list)):
        traj_rel_ = traj_rel[:,i,:].view(-1,2).unsqueeze(dim=1) # obs_len/seq_len x 1 (emulating batch) x 2 (rel-rel)
        curr_first_obs = first_obs[i,:].unsqueeze(dim=0) # 1 x 2

        abs_traj_ = dataset_utils.relative_to_abs(traj_rel_, curr_first_obs).view(-1,2) # "abs" (around 0)
        obj_id = object_class_id_list[i]
        seq_list.append([obj_id,abs_traj_])

        if obj_id == 1: # Agent
            if seq_len == obs_len:
                obs_rel_ = traj_rel_
            elif seq_len > obs_len:
                obs_rel_ = traj_rel_[:obs_len,:,:]

            if len(pred_trajectories_rel.shape) > 1: # != torch.zeros((0))
                if pred_trajectories_rel.shape[1] == 1: # Output from our model (pred_len x 1 x 2 -> Unimodal
                    agent_traj_with_pred_rel = torch.cat((obs_rel_,pred_trajectories_rel),dim=0)
                    agent_traj_with_pred_abs = dataset_utils.relative_to_abs(agent_traj_with_pred_rel, curr_first_obs).view(-1,2)
                    agent_pred_abs = agent_traj_with_pred_abs[obs_len:,:]
                else: # Output from our model (pred_len x N x 2 -> Multimodal)
                    agent_pred_abs = []

                    for t in range(pred_trajectories_rel.shape[1]): # num_modes
                        pred_traj_rel_ = pred_trajectories_rel[:,t,:].unsqueeze(dim=1) # 30 x 1 x 2 (each mode)
                        agent_traj_with_pred_rel = torch.cat((obs_rel_,pred_traj_rel_),dim=0)
                        agent_traj_with_pred_abs = dataset_utils.relative_to_abs(agent_traj_with_pred_rel, curr_first_obs).view(-1,2)
                        agent_pred_abs_ = agent_traj_with_pred_abs[obs_len:,:]

                        agent_pred_abs.append(agent_pred_abs_)

    # Plot all the tracks up till current frame

    for seq_id in seq_list:
        object_type = translate_object_type(int(seq_id[0]))
        seq_abs = seq_id[1]

        if seq_len > obs_len:
            obs_x, obs_y = seq_abs[:obs_len,0] + xcenter, seq_abs[:obs_len,1] + ycenter
            pred_gt_x, pred_gt_y = seq_abs[obs_len:,0] + xcenter, seq_abs[obs_len:,1] + ycenter
        elif seq_len == obs_len:
            obs_x, obs_y = seq_abs[:,0] + xcenter, seq_abs[:,1] + ycenter

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
                    if pred_trajectories_rel.shape[1] == 1: # Unimodal
                        cor_x, cor_y = agent_pred_abs[:,0] + xcenter, agent_pred_abs[:,1] + ycenter
                    else: # Multimodal
                        agent_pred_abs_ = agent_pred_abs[i_mm]
                        cor_x, cor_y = agent_pred_abs_[:,0] + xcenter, agent_pred_abs_[:,1] + ycenter
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

        trajs_folder = os.path.join(*filename.split('/')[:-2],"data_images_trajs/")
        if not os.path.exists(trajs_folder):
            print("Create trajs folder: ", trajs_folder)
            os.makedirs(trajs_folder) # makedirs creates intermediate folders

        if data_aug:
            filename2 = trajs_folder + curr_seq + "_" + str(rot_angle) + "_aug" + ".png"
        elif len(pred_trajectories_rel) != 1:
            filename2 = trajs_folder + curr_seq + f"_qualitative_{mode}" + ".png"
        else:
            filename2 = trajs_folder + curr_seq + "_original" + ".png"

        cv2.imwrite(filename2,full_img_cv)

    resized_full_img_cv = cv2.resize(full_img_cv,(224,224))
    norm_resized_full_img_cv = resized_full_img_cv / 255.0
    
    plt.cla()
    plt.close('all')

    return norm_resized_full_img_cv