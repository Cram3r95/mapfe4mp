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

# Aux functions

COLORS = "darkviolet"
COLORS_MM = ["orange", "chartreuse", "khaki"]

                        # Observation / Prediction (GT) / Prediction (Model)                   
                        # (color, linewidth, type, size, transparency)
MARKER_DICT = {"AGENT":  (("b",6,"o",15,1.0),("c",6,"*",15,0.5),("c",6,"*",15,1.0)),
               "OTHER":  (("g",4,"o",10,1.0),("g",4,"*",10,0.5)),
               "AV":     (("r",4,"o",10,1.0),("m",4,"*",10,1.0))} 

ZORDER = {"AGENT": 3, "OTHER": 3, "AV": 3}

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

def renderize_image(fig_plot, new_shape=(600,600),normalize=True):
    """
    """
    fig_plot.canvas.draw()

    img_cv = cv2.cvtColor(np.asarray(fig_plot.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    img_rsz = cv2.resize(img_cv, new_shape)                
    
    if normalize:
        img_rsz = img_rsz / 255.0 # Normalize from 0 to 1
    return img_rsz

def change_bg_color(img):
    """
    """
    img[np.all(img == (0, 0, 0), axis=-1)] = (255,255,255)

    return img

def generate_img(img_map, img_lanes, qualitative_results_folder, seq_id, t_img):
    """
    """
    
    ## Foreground
    img2gray = cv2.cvtColor(img_lanes,cv2.COLOR_BGR2GRAY)
    ret3,th3 = cv2.threshold(img2gray,253,255,cv2.THRESH_BINARY)
    mask = 255 - th3
    kk = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img2_fg = cv2.bitwise_and(img_lanes,img_lanes,mask=mask)

    ## Background
    
    mask_bg = 255-cv2.erode(cv2.dilate(img_map, kk), kk)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_map,img_map,mask=mask_inv)

    ## Merge
    full_img_cv = cv2.add(img1_bg,img2_fg)
    f_img = change_bg_color(full_img_cv)

    if t_img == 0: # Original image
        filename = qualitative_results_folder + "/" + seq_id + ".png"
    if t_img == 1: # Original image with goal proposals
        filename = qualitative_results_folder + "/" + seq_id + "_goals" + ".png"
    if t_img == 2: # Unimodal prediction
        filename = qualitative_results_folder + "/" + seq_id + "_unimodal" + ".png"
    if t_img == 3: # Multimodal prediction
        filename = qualitative_results_folder + "/" + seq_id + "_multimodal" + ".png"
    cv2.imwrite(filename, f_img)

# Main plot functions

def plot_trajectories(filename,traj_rel,first_obs,map_origin,object_class_id_list,offset,
                      rot_angle=-1,obs_len=20,smoothen=False,save=False,data_aug=False,
                      qualitative_results=False,pred_trajectories_rel=torch.zeros((0))):
    """
    N.B. All inputs must be torch tensors in CPU, not GPU (in order to plot the trajectories)!

    Plot until obs_len points per trajectory. If obs_len != None, we
    must distinguish between observation (color with a marker) and prediction (same color with another marker)
    """

    seq_len = traj_rel.shape[0]
    if seq_len == obs_len:
        plot_len = 1
    elif seq_len > obs_len:
        plot_len = 2

    if len(pred_trajectories_rel.shape) == 3: # Output from our model (pred_len x 1 x 2 -> Unimodal)
                                              #                       (pred_len x num_modes x 2 -> Multimodal)
        plot_len = 3

    xcenter, ycenter = map_origin[0].item(), map_origin[1].item()

    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    plot_object_trajectories = True
    plot_object_heads = True

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Comment if you want to generate images with axis

    t0 = time.time()

    object_type_tracker: Dict[int, int] = defaultdict(int)

    seq_list = []

    for i in range(len(object_class_id_list)):
        traj_rel_ = traj_rel[:,i,:].view(-1,2).unsqueeze(dim=1) # obs_len/seq_len x 1 (emulating batch) x 2 (rel-rel)
        curr_first_obs = first_obs[i,:].unsqueeze(dim=0) # 1 x 2

        abs_traj_ = dataset_utils.relative_to_abs(traj_rel_, curr_first_obs).view(-1,2) # "abs" (around 0)
        obj_id = object_class_id_list[i]
        seq_list.append([obj_id,abs_traj_])

        # if obj_id == 1: # Agent
        #     if seq_len == obs_len:
        #         obs_rel_ = traj_rel_
        #     elif seq_len > obs_len:
        #         obs_rel_ = traj_rel_[:obs_len,:]
        #     pdb.set_trace()
            #traj_with_pred_rel = 
   
    # Plot all the tracks up till current frame

    for seq_id in seq_list:
        object_type = translate_object_type(int(seq_id[0]))
        seq_abs = seq_id[1]

        if seq_len > obs_len:
            obs_x, obs_y = seq_abs[:obs_len,0] + xcenter, seq_abs[:obs_len,1] + ycenter
            pred_gt_x, pred_gt_y = seq_abs[obs_len:,0] + xcenter, seq_abs[obs_len:,1] + ycenter
        elif seq_len == obs_len:
            obs_x, obs_y = seq_abs[:,0] + xcenter, seq_abs[:,1] + ycenter

        if plot_object_trajectories:
            for i in range(plot_len):
                if i == 0: # obs
                    cor_x, cor_y = obs_x, obs_y 
                elif i == 1: # pred (gt)
                    cor_x, cor_y = pred_gt_x, pred_gt_y
                elif i == 2 and object_type == "AGENT": # pred (model) At this moment, only target AGENT
                    cor_x, cor_y = pred_gt_x, pred_gt_y 
                else:
                    continue

                colour,linewidth,marker_type,marker_size,transparency = MARKER_DICT[object_type][i]

                if smoothen:
                    polyline = np.column_stack((cor_x, cor_y))
                    num_points = cor_x.shape[0] * 3
                    smooth_polyline = interpolate_polyline(polyline, num_points)

                    cor_x, cor_y = smooth_polyline[:,0], smooth_polyline[:,1]

                    if cor_x.device.type != "cpu": cor_x = cor_x.to("cpu")
                    if cor_y.device.type != "cpu": cor_y = cor_y.to("cpu")

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
                    plt.scatter(cor_x, cor_y, c=colour, s=10, alpha=transparency)

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

    # print("Time consumed by objects rendering: ", time.time()-t0)

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

    height, width = img_map.shape[:-1]

    ## Foreground

    img_lanes = renderize_image(fig,new_shape=(width,height),normalize=False)
    img2gray = cv2.cvtColor(img_lanes,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,253,255,cv2.THRESH_BINARY_INV)
    img2_fg = cv2.bitwise_and(img_lanes,img_lanes,mask=mask)

    ## Background

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_map,img_map,mask=mask_inv)

    ## Merge

    full_img_cv = cv2.add(img1_bg,img2_fg)

    if save:
        curr_seq = filename.split('/')[-1].split('.')[0]

        trajs_folder = os.path.join(*filename.split('/')[:-2],"data_images_trajs/")
        if not os.path.exists(trajs_folder):
            print("Create trajs folder: ", trajs_folder)
            os.makedirs(trajs_folder) # makedirs creates intermediate folders

        if data_aug:
            filename2 = trajs_folder + curr_seq + "_" + str(rot_angle) + "_aug" + ".png"
        elif qualitative_results: # this would include also the prediction
            filename2 = trajs_folder + curr_seq + "_qualitative" + ".png"
        else:
            filename2 = trajs_folder + curr_seq + "_original" + ".png"

        cv2.imwrite(filename2,full_img_cv)

    resized_full_img_cv = cv2.resize(full_img_cv,(224,224))
    norm_resized_full_img_cv = resized_full_img_cv / 255.0
    
    return norm_resized_full_img_cv

# TODO: REFACTORIZE FROM HERE (MERGE plot_trajectories, plot_qualitative_results and plot_qualitative_results_mm
# in a single function, to observe:
# 1. Only observations (either if only trajs are provided or also the whole trajectory)
# 2. Observations + Unimodal predictions
# 3. Observations + Multimodal predictions

def plot_qualitative_results(filename, pred_traj_fake_list, agent_pred_traj_gt, agent_idx, 
                             object_cls, obs_traj, origin_pos, offset, t_img):
    """
    pred_traj_fake_list may be multimodal
    agent_pred_traj_gt -> main agent
    TODO: Refactorize this function
    """
    img_map = cv2.imread(filename)
    height, width = img_map.shape[:-1]
    # scale = 2
    # img_map = cv2.resize(img_map, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)

    agent_pred_traj_gt = agent_pred_traj_gt.cpu()
    object_cls = object_cls.cpu().numpy()
    obs_traj = obs_traj.cpu()

    plot_object_trajectories = True
    plot_object_heads = True

    color_dict = {"AGENT": (1.0,0.0,0.0,1.0), 
                  "AV": (0.0,0.0,1.0,1.0), 
                  "OTHER": (0.0,0.5,0.0,1.0)} 
    object_type_tracker: Dict[int, int] = defaultdict(int)

    xcenter, ycenter = origin_pos[0][0][0].cpu().item(), origin_pos[0][0][1].cpu().item()
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")

    # Get goal points
    dist_around = 40
    ori_pos = [xcenter,ycenter]

    agent_obs_seq_global = obs_traj[:, agent_idx, :].view(-1,2).numpy() + origin_pos[0][0].cpu().numpy() # Global (HDmap)
    goal_points = dataset_utils.get_goal_points(filename, torch.tensor(agent_obs_seq_global), 
                                                torch.tensor(ori_pos), dist_around,NUM_GOAL_POINTS=32)

    ## radius of action
    vel = dataset_utils.get_agent_velocity(torch.transpose(obs_traj[:, agent_idx, :].view(-1,2),0,1))
    r = 3*vel
    if t_img > 0:
        # circ_car = plt.Circle((xcenter, ycenter), r, color="brown", fill=False, linewidth=3)
        # ax.add_patch(circ_car)
        ax.scatter(goal_points[:,0], goal_points[:,1], marker="x", color='brown', s=8)

    ## abs gt
    agent_pred_traj_gt = agent_pred_traj_gt.view(-1,2).numpy() + origin_pos[0][0].cpu().numpy()

    # Plot observations

    obs_seq_list = []

    for i in range(len(object_cls)):
        abs_obs_ = obs_traj[:,i,:]

        obj_id = object_cls[i]
        obs_seq_list.append([abs_obs_,obj_id])
    
    ag_count = 0
    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)
        c = color_dict[object_type] 

        if object_type == "AGENT":
            marker_type = "*"
            marker_size = 14 #15
            linewidth = 4
            c_m = c
        elif object_type == "OTHER":
            marker_type = "s"
            marker_size = 10 #10
            linewidth = 4
            c_m = c
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 10 # 10
            linewidth = 4
            c_m = c

        cor_x = seq_rel[:,0] + xcenter #+ width/2
        cor_y = seq_rel[:,1] + ycenter #+ height/2

        smoothen = True
        if not smoothen: #
            polyline = np.column_stack((cor_x, cor_y))

            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)

            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        if plot_object_trajectories:
            if smoothen:
                ax.plot(
                    cor_x,
                    cor_y,
                    "-",
                    # color=color_dict[object_type],
                    color=c,
                    label=object_type if not object_type_tracker[object_type] else "",
                    alpha=1,
                    linewidth=linewidth,
                    zorder=_ZORDER[object_type],
                )
                # # draw gt agent and preds
                if object_type == "AGENT":
                    if t_img > 1:
                        ax.plot(
                            agent_pred_traj_gt[:, 0],
                            agent_pred_traj_gt[:, 1],
                            "*",
                            color="orange",
                            label=object_type if not object_type_tracker[object_type] else "",
                            alpha=1,
                            linewidth=0.5,
                            zorder= _ZORDER[object_type],
                        )
                        samples = 1 #len(pred_traj_fake_list)
                        for i in range(samples):
                            pred = pred_traj_fake_list[i]
                            pred = pred.cpu().view(-1,2).numpy() + origin_pos[0][0].cpu().numpy()
                            c = COLORS
                            ax.plot(
                                pred[:, 0],
                                pred[:, 1],
                                "*",
                                color=c,
                                label=object_type if not object_type_tracker[object_type] else "",
                                alpha=1,
                                linewidth=0.5,
                                zorder= _ZORDER[object_type],
                            )
            else:
                ax.scatter(cor_x, cor_y, c=c, s=10)

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if plot_object_heads:
            ax.plot(
                final_x,
                final_y,
                marker_type,
                # color=color_dict[object_type],
                color=c_m,
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
            )

    # Plot our predicted AGENT trajectories (depending on k=num_trajs to be predicted)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_axis_off()

    img_lanes = renderize_image(fig,new_shape=(width,height),normalize=False)

    # Merge information
    split_folder = '/'.join(filename.split('/')[:-2])
    qualitative_results_folder = split_folder + "/qualitative_results_no_circle"
    if not os.path.exists(qualitative_results_folder):
        print("Create qualitative results folder: ", qualitative_results_folder)
        os.makedirs(qualitative_results_folder) # makedirs creates intermediate folders
    
    seq_id = filename.split('/')[-1].split('.')[0]

    # if t_img > 0:
    #     plt.savefig(f"/home/robesafe/shared_home/test_plt/test_{seq_id}_plt_save_fig.png", bbox_inches='tight', facecolor=fig.get_facecolor(), 
    #             edgecolor='none', pad_inches=0)
    #     cv2.imwrite(f"/home/robesafe/shared_home/test_plt/test_{seq_id}_fg.png",img_lanes)
    #     cv2.imwrite(f"/home/robesafe/shared_home/test_plt/test_{seq_id}_bg.png",img_map)

    plt.savefig("aux.png", bbox_inches='tight', facecolor=fig.get_facecolor(), 
                edgecolor='none', pad_inches=0) # TODO: Optimize this -> Return image without white padding instead of saving and reading
    img_lanes = cv2.imread("aux.png")

    generate_img(img_map, img_lanes, qualitative_results_folder, seq_id, t_img=t_img)

def plot_qualitative_results_mm(filename, pred_traj_fake_list, agent_pred_traj_gt, agent_idx, 
                             object_cls, obs_traj, origin_pos, offset):
    """
    pred_traj_fake_list may be multimodal
    agent_pred_traj_gt -> main agent
    TODO: Refactorize this function
    """
    agent_pred_traj_gt = agent_pred_traj_gt.cpu()
    object_cls = object_cls.cpu().numpy()
    obs_traj = obs_traj.cpu()

    plot_object_trajectories = True
    plot_object_heads = True
    color_dict = {"AGENT": (0.0,0.0,1.0,1.0), # BGR
                  "AV": (1.0,0.0,0.0,1.0), 
                  "OTHER": (0.0,1.0,0.0,1.0)} 
    object_type_tracker: Dict[int, int] = defaultdict(int)

    xcenter, ycenter = origin_pos[0][0][0].cpu().item(), origin_pos[0][0][1].cpu().item()
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off")

    # Get goal points
    dist_around = 40
    ori_pos = [xcenter,ycenter]
    agent_obs_seq_global = obs_traj[:, agent_idx, :].view(-1,2).numpy() + origin_pos[0][0].cpu().numpy() # Global (HDmap)
    goal_points = goal_points_functions.get_goal_points(filename, torch.tensor(agent_obs_seq_global), torch.tensor(ori_pos), dist_around)
    
    ## radius of action
    vel = goal_points_functions.get_agent_velocity(torch.transpose(obs_traj[:, agent_idx, :].view(-1,2),0,1))
    r = 3*vel
    circ_car = plt.Circle((xcenter, ycenter), r, color='purple', fill=False, linewidth=3)
    ax.add_patch(circ_car)
    ax.scatter(goal_points[:,0], goal_points[:,1], marker="x", color='purple', s=8)
    
    ## abs gt
    agent_pred_traj_gt = agent_pred_traj_gt.view(-1,2).numpy() + origin_pos[0][0].cpu().numpy()

    # Plot observations

    obs_seq_list = []

    for i in range(len(object_cls)):
        abs_obs_ = obs_traj[:,i,:]

        obj_id = object_cls[i]
        obs_seq_list.append([abs_obs_,obj_id])
    
    ag_count = 0
    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)

        if object_type == "AGENT":
            marker_type = "*"
            marker_size = 14 #15
            linewidth = 4
            c = "b" # blue in rgb (final image)
            c_m = c
        elif object_type == "OTHER":
            marker_type = "s"
            marker_size = 10 #10
            linewidth = 4
            c = "g"
            c_m = c
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 10 # 10
            linewidth = 4
            c = "r" # blue in rgb (final image)
            c_m = c

        cor_x = seq_rel[:,0] + xcenter #+ width/2
        cor_y = seq_rel[:,1] + ycenter #+ height/2

        smoothen = True
        if not smoothen:
            polyline = np.column_stack((cor_x, cor_y))

            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)

            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        if plot_object_trajectories:
            if smoothen:
                plt.plot(
                    cor_x,
                    cor_y,
                    "-",
                    # color=color_dict[object_type],
                    color=c,
                    label=object_type if not object_type_tracker[object_type] else "",
                    alpha=1,
                    linewidth=linewidth,
                    zorder=_ZORDER[object_type],
                )
                # # draw gt agent and preds
                if object_type == "AGENT":
                    plt.plot(
                        agent_pred_traj_gt[:, 0],
                        agent_pred_traj_gt[:, 1],
                        "*",
                        # color=color_dict[object_type],
                        color="aqua",
                        label=object_type if not object_type_tracker[object_type] else "",
                        alpha=1,
                        linewidth=0.5,
                        zorder= _ZORDER[object_type],
                    )
                    samples = len(pred_traj_fake_list)
                    for i in range(samples):
                        preds = pred_traj_fake_list[i].cpu()
                        m = preds.shape[1]
                        for j in range(m):
                            pred = preds[:,j,:,:]
                            pred = pred.view(-1,2).numpy() + origin_pos[0][0].cpu().numpy()
                            c = COLORS_MM[j]
                            plt.plot(
                                pred[:, 0],
                                pred[:, 1],
                                "*",
                                # color=color_dict[object_type],
                                color=c,
                                label=object_type if not object_type_tracker[object_type] else "",
                                alpha=1,
                                linewidth=0.5,
                                zorder= _ZORDER[object_type],
                            )
            else:
                plt.scatter(cor_x, cor_y, c=c, s=10)

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if plot_object_heads:
            plt.plot(
                final_x,
                final_y,
                marker_type,
                # color=color_dict[object_type],
                color=c_m,
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
            )

    # Merge information

    img_map = cv2.imread(filename)
    height, width = img_map.shape[:-1]

    scale = 2

    img_map = cv2.resize(img_map, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)

    # Plot our predicted AGENT trajectories (depending on k=num_trajs to be predicted)
    img_lanes = renderize_image(fig,new_shape=(width*scale,height*scale),normalize=False)

    # Merge information
    generate_img(img_map, img_lanes, filename, t_img=3)