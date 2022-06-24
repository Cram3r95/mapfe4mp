#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 7 12:33:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import random
import math
import pdb
import copy
import copy
import time
import pdb
import os
from collections import defaultdict
from typing import Dict, Tuple, Union

# DL & Math imports

import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

from argoverse.utils.centerline_utils import (
    centerline_to_polygon
)

#######################################

# Aux functions

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
    plt.axis("off") # Comment if you want to generate images with x|y-labels
    
    ## Plot nearby segments

    # avm.plot_nearby_halluc_lanes(ax, city_name, xcenter, ycenter)

    ## Plot hallucinated polygones and centerlines

    t0 = time.time()

    for lane_cl in lane_centerlines:
        lane_polygon = centerline_to_polygon(lane_cl[:, :2])
                                                                          #"black"  
        ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "white", edgecolor='white', fill=True)
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "-", color="grey", linewidth=1.5, alpha=1.0, zorder=2)
        # N.B. zorder must be 2 (Line2D) in order to represent the whole line, with 1 (Path) there are some problems

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