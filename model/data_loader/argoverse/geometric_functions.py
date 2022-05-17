#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Geometric functions

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

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    try:
        x,y,z = b.x, b.y, 0.0
        X,Y,Z = e.x, e.y, 0.0
    except:
        x,y,z = b
        X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v

    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    try:   
        X,Y,Z = w.x, w.y, 0.0
    except:
        X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)

    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def get_non_linear(file_id, curr_seq, idx=0, obj_kind=2, threshold=2, debug_trajectory_classifier=False):
    """
    Non-linear means the trajectory (of the AGENT in the present case) is a curve. 
    Otherwise, it is considered as a linear (straight) trajectory
    """

    agent_seq = curr_seq[idx,:,:] #.cpu().detach().numpy()
    num_points = curr_seq.shape[2]

    agent_x = agent_seq[0,:].reshape(-1,1)
    agent_y = agent_seq[1,:].reshape(-1,1)

    # Fit a RANSAC regressor  

    ransac = linear_model.RANSACRegressor(residual_threshold=threshold, 
                                          max_trials=30, 
                                          min_samples=round(0.6*num_points))
    ransac.fit(agent_x,agent_y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    num_inliers = len(np.where(inlier_mask == True)[0])

    ## Study consecutive outliers

    cnt = 0
    num_min_outliers = 8 # Minimum number of consecutive outliers to consider the trajectory as curve
    non_linear = 0.0
    reason = ""
    for is_inlier in inlier_mask:
        if not is_inlier:
            cnt += 1
        else:
            cnt = 0

        if cnt >= num_min_outliers:
            non_linear = 1.0
            reason = "ransac"
            break

    # Study distance from intermediate points to the line that links the first and last point

    first_point = (agent_x[0], agent_y[0], 0)
    last_point = (agent_x[-1], agent_y[-1], 0)
    num_out = 0
    num_far_out = 0
    num_close_out = 0
    flag = False

    for index in range(num_points):
        point = (agent_x[index], agent_y[index], 0)
        dist,_ = pnt2line(point,first_point,last_point)
        # print("index, dist: ", index, dist)
        if dist >= threshold:  
            num_out += 1
            if num_out >= num_min_outliers:
                flag = True
                reason = reason + " + normal outliers"
                break

            if dist >= threshold*1.5:
                num_far_out += 1
                if num_far_out >= round(0.5 * num_min_outliers):
                    flag = True
                    reason = reason + " + far outliers"
                    break
        if dist >= round(0.66*threshold):
            num_close_out += 1
            if num_close_out >= round(1.2 * num_min_outliers):
                flag = True
                reason = reason + " + close outliers"
                break
        else:
            num_out = 0
            num_far_out
            num_close_out = 0

    # coef, res, _, _, _ = np.polyfit(agent_x.flatten(),agent_y.flatten(),1,full=True)
    # coef, res_list = np.polynomial.hermite.hermfit(agent_x.flatten(),agent_y.flatten(),2,full=True)
    # res = res_list[0]

    if non_linear or flag:
        non_linear = 1.0
    else:
        non_linear = 0.0

    if debug_trajectory_classifier:
        x_max = agent_x.max()
        x_min = agent_x.min()
        num_steps = 20
        step_dist = (x_max - x_min) / num_steps
        line_x = np.arange(x_min, x_max, step_dist)[:, np.newaxis]
        line_y_ransac = ransac.predict(line_x)

        y_min = line_y_ransac.min()
        y_max = line_y_ransac.max()
        
        plt.scatter(
            agent_x[inlier_mask], agent_y[inlier_mask], color="blue", marker=".", label="Inliers"
        )
        plt.scatter(
            agent_x[outlier_mask], agent_y[outlier_mask], color="red", marker=".", label="Outliers"
        )
        plt.scatter(agent_x[0], agent_y[0], s=200, color='green', marker=".", label="First obs")

        lw = 2
        plt.plot(
            line_x,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=lw,
            label="RANSAC regressor",
        )

        # yfit = np.polyval(coef,line_x)
        # plt.plot(line_x,yfit, label='fit')

        plt.legend(loc="lower right")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if non_linear == 1.0:
            non_linear = True
        else:
            non_linear = False

        if obj_kind == 0:
            obj = "AV"
        elif obj_kind == 1:
            obj = "AGENT"
        else:
            obj = "OTHER"

        plt.title('Sequence {}. Obstacle: {}. Num inliers: {}. Is a curve: {}. Reason: {}'.format(file_id,obj,num_inliers,non_linear,reason))

        threshold = 5
        plt.xlim([x_min-threshold, x_max+threshold])
        plt.ylim([y_min-threshold, y_max+threshold])

        plt.show()
    return non_linear

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0