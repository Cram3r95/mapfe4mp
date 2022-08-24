#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Goal points functions

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import random
import math
import pdb
import copy

# DL & Math imports

import numpy as np
import torch
import cv2
from sklearn import linear_model

# Plot imports

import matplotlib.pyplot as plt

#######################################

def get_points(img, car_px, scale_x, rad=100, color=255, N=1024, sample_car=True, max_samples=None):
    """
    """

    feasible_area = np.where(img == color)
    sampling_points = np.vstack(feasible_area)

    rng = np.random.default_rng()
    num_samples = N
    
    points_feasible_area = len(feasible_area[0])
    # num_samples = int(points_feasible_area/2)
    sample_index = rng.choice(points_feasible_area, size=N, replace=False)

    sampling_points = sampling_points[:,sample_index]
    
    px_y = sampling_points[0,:] # rows (pixels)
    px_x = sampling_points[1,:] # columns (pixels)
    
    ## sample points in the car radius
    
    if sample_car:
        final_points = [[a,b] for a,b in zip(px_y,px_x) if (math.sqrt(pow(a - car_px[0],2)+
                                                                      pow(b - car_px[1],2)) < rad)]

        if max_samples:               
            final_points = random.sample(final_points,max_samples)
            assert len(final_points) == max_samples
          
        final_points = np.array(final_points)

        try:
            px_y = final_points[:,0] # rows
            px_x = final_points[:,1] # columns
        except:
            scale_y = scale_x
            px_y = car_px[0] + scale_y*np.random.randn(num_samples) # columns
            px_x = car_px[1] + scale_x*np.random.randn(num_samples) # rows
                  
    return px_y, px_x

def change_bg_color(img):
    img[np.all(img == (0, 0, 0), axis=-1)] = (255,255,255)

    return img

def get_agent_acceleration(obs_seq, period=0.1):
    """
    """

    vel = np.zeros((obs_seq.shape[1]-1))

    for i in range(1,obs_seq.shape[1]):
        x_pre, y_pre = obs_seq[:,i-1]
        x_curr, y_curr = obs_seq[:,i]

        dist = math.sqrt(pow(x_curr-x_pre,2)+pow(y_curr-y_pre,2))

        curr_vel = dist / period
        vel[i-1] = curr_vel

    print("vel: ", vel)
    print("Vel: ", vel.mean())
    d = vel.mean() * 3
    print("m using CTRV: ", d)

    acc = np.zeros((len(vel)-1))

    for i in range(1,len(vel)):
        vel_pre = vel[i-1]
        vel_curr = vel[i]

        delta_vel = vel_curr - vel_pre

        curr_acc = delta_vel / period
        acc[i-1] = curr_acc

    cte_acc = acc.mean()

    d = vel[-1]*3 + 1/2*cte_acc*3**2
    print("accs: ", acc)
    print("ACC: ", cte_acc)
    print("m using CTRA: ", d)

    return cte_acc

def get_agent_velocity(obs_seq, num_obs=5, period=0.1):
    """
    Consider the last num_obs points to calculate an average velocity of
    the object in the last observation point
    TODO: Consider an acceleration?
    """

    obs_seq_vel = obs_seq[:,-5:]

    vel = np.zeros((num_obs-1))

    for i in range(1,obs_seq_vel.shape[1]):
        x_pre, y_pre = obs_seq_vel[:,i-1]
        x_curr, y_curr = obs_seq_vel[:,i]

        dist = math.sqrt(pow(x_curr-x_pre,2)+pow(y_curr-y_pre,2))

        curr_vel = dist / period
        vel[i-1] = curr_vel

    return vel.mean()

def get_agent_yaw(obs_seq, num_obs=5):
    """
    Consider the last num_obs points to calculate an average yaw of
    the object in the last observation point
    """

    obs_seq_yaw = obs_seq[:,-5:]

    yaw = np.zeros((num_obs-1))

    for i in range(1,obs_seq_yaw.shape[1]):
        x_pre, y_pre = obs_seq_yaw[:,i-1]
        x_curr, y_curr = obs_seq_yaw[:,i]

        delta_y = y_curr - y_pre
        delta_x = x_curr - x_pre
        curr_yaw = math.atan2(delta_y, delta_x)

        yaw[i-1] = curr_yaw

    yaw = yaw[np.where(yaw != 0)]
    if len(yaw) == 0: # All angles were 0
        yaw = np.zeros((1))

    tolerance = 0.1
    num_positives = len(np.where(yaw > 0)[0])
    num_negatives = len(yaw) - num_positives
    final_yaw = yaw.mean()

    if (yaw.std() > 1.5): # and ((yaw.mean() > math.pi * (1 - tolerance) and yaw.mean() < - math.pi * (1 - tolerance)) # Around pi
                          # or (yaw.mean() > -math.pi/12 * (1 - tolerance) and yaw.mean() < math.pi/12 * (1 - tolerance)))): # Around 0
        yaw = np.absolute(yaw)

        if num_negatives > num_positives:
            final_yaw = -yaw.mean()
        else:
            final_yaw = yaw.mean()

    return final_yaw

def transform_px2real_world(px_points, origin_pos, real_world_offset, img_size):
    """
    It is assumed squared image (e.g. 600 x 600 -> img_size = 600) and the same offset 
    in all directions (top, bottom, left, right) to facilitate the transformation.
    """

    xcenter, ycenter = origin_pos[0], origin_pos[1]
    x_min = xcenter - real_world_offset
    x_max = xcenter + real_world_offset
    y_min = ycenter - real_world_offset
    y_max = ycenter + real_world_offset

    m_x = float((2 * real_world_offset) / img_size) # slope
    m_y = float(-(2 * real_world_offset) / img_size) # slope

    i_x = x_min # intersection
    i_y = y_max

    rw_points = []

    for px_point in px_points:
        x = m_x * px_point[1] + i_x # Get x-real_world from columns
        y = m_y * px_point[0] + i_y # Get y-real_world from rows
        rw_point = [x,y]
        rw_points.append(rw_point)

    return np.array(rw_points)

def transform_real_world2px(rw_points, origin_pos, real_world_offset, img_size):
    """
    It is assumed squared image (e.g. 600 x 600 -> img_size = 600) and the same offset 
    in all directions (top, bottom, left, right) to facilitate the transformation.
    """

    xcenter, ycenter = origin_pos[0], origin_pos[1]
    x_min = xcenter - real_world_offset
    x_max = xcenter + real_world_offset
    y_min = ycenter - real_world_offset
    y_max = ycenter + real_world_offset

    m_x = float(img_size / (2 * real_world_offset)) # slope
    m_y = float(-img_size / (2 * real_world_offset))

    i_x = float(-(img_size / (2 * real_world_offset)) * x_min) # intercept
    i_y = float((img_size / (2 * real_world_offset)) * y_max)

    px_points = []

    for rw_point in rw_points:
        x = m_x * rw_point[0] + i_x
        y = m_y * rw_point[1] + i_y
        px_point = [x,y] 
        px_points.append(px_point)

    return np.array(px_points)

def get_goal_points(filename, obs_seq, origin_pos, real_world_offset, NUM_GOAL_POINTS=32):
    """
    """

    split_folder = '/'.join(filename.split('/')[:-2])
    goal_points_folder = split_folder + "/goal_points"
    seq_id = filename.split('/')[-1].split('.')[0]

    # 0. Load image and get past observations

    img = cv2.imread(filename)
    img = cv2.resize(img, dsize=(600,600))
    height, width = img.shape[:2]
    img_size = height
    scale_x = scale_y = float(height/(2*real_world_offset))

    cx = int(width/2)
    cy = int(height/2)
    car_px = (cy,cx)

    ori = obs_seq[-1, :]

    # 0. Plot obs traj (AGENT)

    obs_x = obs_seq[:,0]
    obs_y = obs_seq[:,1]

    obs_px_points = transform_real_world2px(obs_seq, origin_pos, real_world_offset, img_size)
    agent_obs_px_x, agent_obs_px_y = obs_px_points[:,0], obs_px_points[:,1]
    filename = goal_points_folder + "/" + seq_id + "_obs_traj.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, change_bg=True)

    # 1. Get feasible area points (N samples)

    # 1.0. (Optional) Observe random sampling in the whole feasible area

    fe_y, fe_x = get_points(img, car_px, scale_x, rad=10000, color=255, N=1024, 
                            sample_car=True, max_samples=None) # return rows, columns
    filename = goal_points_folder + "/" + seq_id + "_all_samples.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, goals_px_x=fe_x, goals_px_y=fe_y, change_bg=True)

    # 1.1. Filter using AGENT estimated velocity

    mean_vel = get_agent_velocity(torch.transpose(obs_seq,0,1))
    pred_seconds = 3 # instead of 3 s (prediction in ARGOVERSE)
    radius = mean_vel * pred_seconds
    radius_px = radius * scale_x	

    fe_y, fe_x = get_points(img, car_px, scale_x, rad=radius_px, color=255, N=1024, 
                                sample_car=True, max_samples=None) # return rows, columns

    filename = goal_points_folder + "/" + seq_id + "_vel_filter.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=fe_x, goals_px_y=fe_y, radius=radius_px, change_bg=True)

    # pdb.set_trace()

    # 1.2. Filter points applying rotation

    mean_yaw = get_agent_yaw(torch.transpose(obs_seq,0,1)) # radians

    if mean_yaw >= 0.0:
        angle = math.pi/2 - mean_yaw
    elif mean_yaw < 0.0:
        angle = -(math.pi / 2 + (math.pi - abs(mean_yaw)))

    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c,-s], [s, c]])

    fe_x_trans = fe_x - cx # get px w.r.t. the center of the image to be rotated
    fe_y_trans = fe_y - cy

    close_pts = np.hstack((fe_x_trans.reshape(-1,1),fe_y_trans.reshape(-1,1)))
    close_pts_rotated = np.matmul(close_pts,R).astype(np.int32)

    fe_x_rot = close_pts_rotated[:,0] + cx
    fe_y_rot = close_pts_rotated[:,1] + cy

    filtered_fe_x = fe_x[np.where(fe_y_rot < cy)[0]]
    filtered_fe_y = fe_y[np.where(fe_y_rot < cy)[0]]

    filename = goal_points_folder + "/" + seq_id + "_angle_filter.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=filtered_fe_x, goals_px_y=filtered_fe_y, radius=radius_px, change_bg=True)

    # 2. Get furthest N samples (closest the the hypothetical radius)

    dist = []
    for i in range(len(filtered_fe_x)):
        d = math.sqrt(pow(filtered_fe_x[i] - car_px[0],2) + pow(filtered_fe_y[i] - car_px[1],2))
        dist.append(d)

    dist = np.array(dist)

    furthest_indeces = np.argsort(dist)
    if len(furthest_indeces) > NUM_GOAL_POINTS:
        furthest_indeces = np.argsort(dist)[-NUM_GOAL_POINTS:]

    final_samples_x, final_samples_y = filtered_fe_x[furthest_indeces], filtered_fe_y[furthest_indeces]
    
    try:
        diff_points = NUM_GOAL_POINTS - len(final_samples_x)
        final_samples_x = np.hstack((final_samples_x, final_samples_x[0]+0.2 * np.random.randn(diff_points)))
        final_samples_y = np.hstack((final_samples_y, final_samples_y[0]+0.2 * np.random.randn(diff_points)))
    except:
        final_samples_x = cx + scale_x*np.random.randn(NUM_GOAL_POINTS)
        final_samples_y = cy + scale_y*np.random.randn(NUM_GOAL_POINTS)

    filename = goal_points_folder + "/" + seq_id + "_final_samples.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=final_samples_x, goals_px_y=final_samples_y, radius=radius_px, change_bg=True)

    if len(final_samples_x) != NUM_GOAL_POINTS:
        print(f"Final samples does not match with {NUM_GOAL_POINTS} required samples")
        # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
        #               goals_px_x=final_samples_x, goals_px_y=final_samples_y, radius=radius_px, change_bg=True, show=True)
        pdb.set_trace()

    # 3. Transform pixels to real-world coordinates

    final_samples_px = np.hstack((final_samples_y.reshape(-1,1), final_samples_x.reshape(-1,1))) # rows, columns
    rw_points = transform_px2real_world(final_samples_px, origin_pos, real_world_offset, img_size)
    # pdb.set_trace()
    return rw_points

# N.B. In PLT, points must be specified as standard cartesian frames (x from left to right, y from bottom to top)
def plot_fepoints(img, filename, seq_px_x, seq_px_y, last_obs_px, obs_origin, 
                  goals_px_x=None, goals_px_y=None, label=None,
                  radius=None, change_bg=False, show_pred_gt=False, show=False, save_fig=False, final_clusters=False):
    assert len(img.shape) == 3
    
    img_aux = copy.deepcopy(img)
    fig, ax = plt.subplots(figsize=(8, 8))

    obs_px_x, obs_px_y = seq_px_x[:obs_origin], seq_px_y[:obs_origin]

    if show_pred_gt:
        plt.scatter(seq_px_x[obs_origin:], seq_px_y[obs_origin:], c="cyan", marker=6, s=50) # GT future trajectory
        
    plt.scatter(obs_px_x, obs_px_y, c="blue", marker=".", s=50) # Past trajectory
    plt.scatter(last_obs_px[0], last_obs_px[1], c="blue", marker="*", s=100) # Last observation point
    
    if goals_px_x is not None:
        if label is not None:
            u_labels = np.unique(label) # get unique labels
            for i in u_labels:
                if final_clusters: goal_size=80
                else: goal_size = 10
                plt.scatter(goals_px_x[label == i] , goals_px_y[label == i] , 
                            label = i, marker="8", s=goal_size) # Goal points clustered
        else:
            plt.scatter(goals_px_x, goals_px_y, color="purple", marker="8", s=10) # Goal points

    plt.imshow(img_aux)

    if radius:
      circ_car = plt.Circle((last_obs_px[0], last_obs_px[1]), radius, color="purple", fill=False)
      ax.add_patch(circ_car)

    plt.axis("off")

    if show:
        plt.title(filename) 
        plt.show()

    if change_bg:
        img_aux = change_bg_color(img)

    if save_fig:
        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), 
                    edgecolor='none', pad_inches=0)

    plt.close('all')