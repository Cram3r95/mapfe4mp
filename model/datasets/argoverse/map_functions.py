#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 7 12:33:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
import copy
import logging
import sys
import time
import pdb
import os

from collections import defaultdict
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

from argoverse.utils.centerline_utils import (
    centerline_to_polygon
)

import model.datasets.argoverse.dataset_utils as dataset_utils
import model.datasets.argoverse.goal_points_functions as goal_points_functions

IS_OCCLUDED_FLAG = 100
LANE_TANGENT_VECTOR_SCALING = 4
plot_lane_tangent_arrows = True

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Aux functions

def renderize_image(fig_plot, new_shape=(600,600),normalize=True):
    fig_plot.canvas.draw()

    img_cv = cv2.cvtColor(np.asarray(fig_plot.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    img_rsz = cv2.resize(img_cv, new_shape)                
    
    if normalize:
        img_rsz = img_rsz / 255.0 # Normalize from 0 to 1
    return img_rsz

# _ZORDER = {"AGENT": 15, "AV": 10, "OTHER": 5}
_ZORDER = {"AGENT": 3, "AV": 3, "OTHER": 3}
COLORS = "darkviolet"
COLORS_MM = ["orange", "chartreuse", "khaki"]

def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
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
    if int_id == 0:
        return "AV"
    elif int_id == 1:
        return "AGENT"
    else:
        return "OTHER"

def draw_lane_polygons(
    ax: plt.Axes,
    lane_polygons: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linewidth:float,
    fill: bool,
) -> None:
    """Draw a lane using polygons.
    Args:
        ax: Matplotlib axes
        lane_polygons: Array of (N,) objects, where each object is a (M,3) array
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """

    for i, polygon in enumerate(lane_polygons):
        if fill:
            ax.fill(polygon[:, 0], polygon[:, 1], "black", edgecolor='w', fill=True)
        else:
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth, alpha=1.0, zorder=1)

# Main function for map generation

def map_generator(curr_num_seq,
                  origin_pos,
                  offset,
                  avm,
                  city_name,
                  show: bool = False,
                  root_folder = "data/datasets/argoverse/motion-forecasting/train/data_images") -> None:
    """
    """

    if not os.path.exists(root_folder):
        print("Create experiment path: ", root_folder)
        os.mkdir(root_folder)

    plot_centerlines = True
    plot_local_das = False
    plot_local_lane_polygons = False

    xcenter, ycenter = origin_pos[0], origin_pos[1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    # Get map info 

    ## Get centerlines around the origin

    t0 = time.time()

    seq_lane_props = dict()
    if plot_centerlines:
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    ### Get lane centerlines which lie within the range of trajectories

    lane_centerlines = []
    
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min):

            lane_centerlines.append(lane_cl)

    ## Get local polygons around the origin

    local_lane_polygons = []
    if plot_local_lane_polygons:
        local_lane_polygons = avm.find_local_lane_polygons([x_min, 
                                                            x_max, 
                                                            y_min, 
                                                            y_max], 
                                                            city_name)
    #pdb.set_trace()
    ## Get drivable area around the origin

    local_das = []
    if plot_local_das:
        local_das = avm.find_local_driveable_areas([x_min, 
                                                    x_max, 
                                                    y_min, 
                                                    y_max], 
                                                    city_name)

    # print("\nTime consumed by local das and polygons calculation: ", time.time()-t0)

    # Plot

    fig, ax = plt.subplots(figsize=(6,6), facecolor="black")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Comment if you want to generate images without x|y-labels
    
    ## Plot nearby segments

    # avm.plot_nearby_halluc_lanes(ax, city_name, xcenter, ycenter)

    ## Plot hallucinated polygones and centerlines

    t0 = time.time()

    for lane_cl in lane_centerlines:
        lane_polygon = centerline_to_polygon(lane_cl[:, :2])
                                                                          #"black"  
        ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "white", edgecolor='white', fill=True)
                                                        #"grey"
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "-", color="black", linewidth=1.5, alpha=1.0, zorder=1)

    # print("Time consumed by plot drivable area and lane centerlines: ", time.time()-t0)

    # draw_lane_polygons(ax, local_das, "tab:pink", linewidth=1.5, fill=False)
    # draw_lane_polygons(ax, local_lane_polygons, "tab:red", linewidth=1.5, fill=False)

    # Save image

    filename = root_folder + "/" + str(curr_num_seq) + ".png"

    if show:
        plt.show()
        pdb.set_trace()

    # Save PLT canvas -> png to automatically remove padding

    plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), 
                edgecolor='none', pad_inches=0)

# Plot trajectories for data augmentation testing

def plot_trajectories(filename,obs_seq,first_obs,origin_pos, object_class_id_list,offset,\
                      rot_angle=-1,obs_len=None,smoothen=False,show=False):
    """
    Plot until plot_len points per trajectory. If plot_len != None, we
    must distinguish between observation (color with a marker) and prediction (same color with another marker)
    """

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    plot_object_trajectories = True
    plot_object_heads = True

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Uncomment if you want to generate images with

    t0 = time.time()

    color_dict = {"AGENT": (0.0,0.0,1.0,1.0), # BGR
                  "AV": (1.0,0.0,0.0,1.0), 
                  "OTHER": (0.0,1.0,0.0,1.0)} 
    object_type_tracker: Dict[int, int] = defaultdict(int)

    obs_seq_list = []
    
    # Filter objects if you are debugging (not running the whole dataloader)
    # Probably you will have 0s in obj class list, in addition to the first 0 (which represents
    # the AV)
    try:
        if len(np.where(object_class_id_list == 0)[0]) > 1:
            start_dummy = np.where(object_class_id_list == 0)[0][1]
            object_class_id_list = object_class_id_list[:start_dummy]
    except Exception as e:
        pdb.set_trace()

    for i in range(len(object_class_id_list)):
        obs_ = obs_seq[:obs_len,i,:].view(-1,2) # 20 x 2 (rel-rel)
        curr_first_obs = first_obs[i,:].view(-1)

        abs_obs_ = dataset_utils.relative_to_abs(obs_, curr_first_obs) # "abs" (around 0)
        obj_id = object_class_id_list[i]
        obs_seq_list.append([abs_obs_,obj_id])

    # Plot all the tracks up till current frame

    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)

        if object_type == "AGENT":
            marker_type = "*"
            marker_size = 15
            linewidth = 6
            c = "b" # blue in rgb (final image)
        elif object_type == "OTHER":
            marker_type = "o"
            marker_size = 10
            linewidth = 4
            c = "g"
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 10
            linewidth = 4
            c = "r" # blue in rgb (final image)

        cor_x = seq_rel[:,0] + xcenter #+ width/2
        cor_y = seq_rel[:,1] + ycenter #+ height/2

        if smoothen:
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
                color=c,
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
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
    elif rot_angle == 270:
        img_map = cv2.rotate(img_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: # -1 (default)
        rot_angle = 0
    height, width = img_map.shape[:-1]

    ## Foreground

    img_lanes = renderize_image(fig,new_shape=(width,height),normalize=False)
    img2gray = cv2.cvtColor(img_lanes,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,127,255,cv2.THRESH_BINARY_INV)
    img2_fg = cv2.bitwise_and(img_lanes,img_lanes,mask=mask)

    ## Background

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_map,img_map,mask=mask_inv)

    ## Merge

    full_img_cv = cv2.add(img1_bg,img2_fg)

    if show:
        # cv2.imshow("full_img",full_img_cv)
        curr_seq = filename.split('/')[-1].split('.')[0]
        filename2 = os.path.join(*filename.split('/')[:-2]) + "/data_images_augs/" + curr_seq + "_" + str(rot_angle) + ".png"
        cv2.imwrite(filename2,full_img_cv)

    resized_full_img_cv = cv2.resize(full_img_cv,(224,224))
    norm_resized_full_img_cv = resized_full_img_cv / 255.0

    return norm_resized_full_img_cv

def change_bg_color(img):
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