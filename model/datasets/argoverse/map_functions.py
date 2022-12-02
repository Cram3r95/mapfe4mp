#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 7 12:33:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

# General purpose imports

import time
import pdb
import os
import git
import copy

from typing import Any, Dict, List, Tuple, Union
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

# DL & Math imports

import math
import torch
import cv2
import scipy as sp
import numpy as np

from scipy.signal import savgol_filter

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

import argoverse.utils.centerline_utils as centerline_utils

from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)

#######################################

# https://matplotlib.org/stable/api/markers_api.html
# https://matplotlib.org/stable/gallery/color/named_colors.html

# N.B. When you plot your visualization, it will be plotted in RGB (so, saved in RGB)
# but if you read that image using CV2, you will need to convert from BGR to RGB

# E.g. 

# img = cv2.imread(img_filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir

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
                  dist_rasterized_map,
                  avm,
                  city_name,
                  centerlines_colour="gray",
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
    x_min = xcenter + dist_rasterized_map[0]
    x_max = xcenter + dist_rasterized_map[1]
    y_min = ycenter + dist_rasterized_map[2]
    y_max = ycenter + dist_rasterized_map[3]

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
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "-", color=centerlines_colour, linewidth=1.5, alpha=1.0, zorder=2)
        # N.B. If you want to represent the centerline on top of the filled polygon, the zorder must be higher

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

# Map Feature computations

_MANHATTAN_THRESHOLD = 10.0  # meters
_DFS_THRESHOLD_FRONT_SCALE = 45.0 # 45.0  # m/s
_DFS_THRESHOLD_BACK_SCALE = 40.0 # 40.0  # m/s
_MAX_SEARCH_RADIUS_CENTERLINES = 10.0  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 10

class MapFeaturesUtils:
    """Utils for computation of map-based features."""
    def __init__(self):
        """Initialize class."""
        self._MANHATTAN_THRESHOLD = _MANHATTAN_THRESHOLD
        self._DFS_THRESHOLD_FRONT_SCALE = _DFS_THRESHOLD_FRONT_SCALE
        self._DFS_THRESHOLD_BACK_SCALE = _DFS_THRESHOLD_BACK_SCALE
        self._MAX_SEARCH_RADIUS_CENTERLINES = _MAX_SEARCH_RADIUS_CENTERLINES
        self._MAX_CENTERLINE_CANDIDATES_TEST = _MAX_CENTERLINE_CANDIDATES_TEST

    def get_point_in_polygon_score(self, lane_seq: List[int],
                                   xy_seq: np.ndarray, city_name: str,
                                   avm: ArgoverseMap) -> int:
        """Get the number of coordinates that lie insde the lane seq polygon.
        Args:
            lane_seq: Sequence of lane ids
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            point_in_polygon_score: Number of coordinates in the trajectory that lie within the lane sequence
        """
        lane_seq_polygon = unary_union([
            Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
            for lane in lane_seq
        ])
        point_in_polygon_score = 0
        for xy in xy_seq:
            point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
        return point_in_polygon_score

    def sort_lanes_based_on_point_in_polygon_score(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
    ) -> List[List[int]]:
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score
        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                self.get_point_in_polygon_score(lane_seq, xy_seq, city_name,
                                                avm))
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores

    def get_heuristic_centerlines_for_test_set(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            max_candidates: int,
            scores: List[int],
    ) -> List[np.ndarray]:
        """Sort based on distance along centerline and return the centerlines.
        
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score 
        """
        aligned_centerlines = []
        diverse_centerlines = []
        diverse_scores = []
        num_candidates = 0

        # Get first half as aligned centerlines
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = avm.get_cl_from_lane_seq([lane_seq], city_name)[0]
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)
        test_centerlines = aligned_centerlines
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]
            test_centerlines += diverse_centerlines

        return test_centerlines

    def get_agent_velocity_and_acceleration(self, agent_seq, obs_len=20, period=0.1, 
                                            upsampling_factor=2, filter=None, debug=False):
        """
        Consider the observation data to calculate an average velocity and 
        acceleration of the agent in the last observation point
        """

        # https://en.wikipedia.org/wiki/Speed_limits_in_the_United_States_by_jurisdiction
        # 1 miles per hour (mph) ~= 1.609 kilometers per hour (kph)
        # Common speed limits in the USA (in highways): 70 - 80 mph -> (112 kph - 129 kph) -> (31.29 m/s - 35.76 m/s)
        # The average is around 120 kph -> 33.33 m/s, so if the vehicle has accelerated strongly (in the last observation has reached the 
        # maximum velocity), we assume the GT will be without acceleration, that is, braking or stop accelerating positively 
        # (e.g. Seq 188893 in train). 
        # The maximum prediction horizon should be around 100 m.

        #                            | (Limit of observation)
        #                            v
        # ... . . .  .  .  .    .    .    .    .    .     .     .     .     .     .     .     .     . (Good interpretation)

        # [     Observation data     ][                       Groundtruth data                      ]  

        # ... . . .  .  .  .    .    .    .     .     .      .       .       .       .       .        .        .          . (Wrong interpretation) 

        x = agent_seq[:,0]
        y = agent_seq[:,1]
        xy = np.vstack((x, y))

        extended_xy_f = np.array([])
        num_points_trajectory = agent_seq.shape[0]

        if filter == "savgol":
            xy_f = savgol_filter(xy, window_length=int(num_points_trajectory/4), polyorder=3, axis=1)
        elif filter == "cubic_spline":
            points = np.arange(agent_seq.shape[0])
            upsampled_num_points_ = upsampling_factor*num_points_trajectory # We upsample the frequency to estimate more
                                                                            # points from the given observation data
            period = period / upsampling_factor

            upsampled_points = np.linspace(points.min(), points.max(), upsampled_num_points_)

            up_x = sp.interpolate.interp1d(points,x,kind='cubic')(upsampled_points)
            up_y = sp.interpolate.interp1d(points,y,kind='cubic')(upsampled_points)

            xy_f = np.vstack((up_x,up_y))

            num_points_trajectory = upsampled_num_points_
        elif filter == "savgol+cubic_spline":
            points = np.arange(num_points_trajectory)
            upsampled_num_points_ = upsampling_factor*num_points_trajectory # We upsample the frequency to estimate more
            # points from the given observation data
            period = period / upsampling_factor

            upsampled_points = np.linspace(points.min(), points.max(), upsampled_num_points_)

            up_x = sp.interpolate.interp1d(points,x,kind='cubic')(upsampled_points)
            up_y = sp.interpolate.interp1d(points,y,kind='cubic')(upsampled_points)

            num_points_trajectory = upsampled_num_points_

            xy_f = np.vstack((up_x,up_y))
            xy_f = savgol_filter(xy_f, window_length=int(num_points_trajectory/4), polyorder=3, axis=1)
        elif filter == "least_squares":
            polynomial_order = 2
            # print("Polynomial order: ", polynomial_order)
            t = np.linspace(1, num_points_trajectory, num_points_trajectory)
            px = np.poly1d(np.polyfit(t,x,polynomial_order))
            py = np.poly1d(np.polyfit(t,y,polynomial_order))

            xy_f = np.vstack((px(t),py(t)))

            seq_len = 50
            t2 = np.linspace(1, seq_len, seq_len)
            extended_xy_f = np.vstack((px(t2),py(t2)))
        else: # No filter, original data
            xy_f = xy

        obs_seq_f = xy_f

        vel_f = np.zeros((num_points_trajectory-1))
        for i in range(1,obs_seq_f.shape[1]):
            x_pre, y_pre = obs_seq_f[:,i-1]
            x_curr, y_curr = obs_seq_f[:,i]

            dist = math.sqrt(pow(x_curr-x_pre,2)+pow(y_curr-y_pre,2))

            curr_vel = dist / period
            vel_f[i-1] = curr_vel
        
        acc_f = np.zeros((num_points_trajectory-2))
        for i in range(1,len(vel_f)):
            vel_pre = vel_f[i-1]
            vel_curr = vel_f[i]

            delta_vel = vel_curr - vel_pre

            curr_acc = delta_vel / period
            acc_f[i-1] = curr_acc

        min_weight = 1
        max_weight = 4

        if filter == "least_squares":
            # Theoretically, if the points are computed using Least Squares with a Polynomial with order >= 2,
            # the velocity in the last observation should be fine, since you are taking into account the 
            # acceleration (either negative or positive)

            vel_f_averaged = vel_f[-1]
        else:
            vel_f_averaged = np.average(vel_f,weights=np.linspace(min_weight,max_weight,len(vel_f))) 

        acc_f_averaged = np.average(acc_f,weights=np.linspace(min_weight,max_weight,len(acc_f)))
        acc_f_averaged_aux = acc_f_averaged
        if vel_f_averaged > 35 and acc_f_averaged > 5:
            acc_f_averaged = 0 # The vehicle cannot drive faster in this problem!
                               # This is an assumption!

        if debug:
            print("Filter: ", filter)
            # print("xy_f: ", xy_f)
            print("Min weight, max weight: ", min_weight, max_weight)
            # print("Vel: ", vel_f)
            print("vel averaged: ", vel_f_averaged)
            # print("Acc: ", acc_f)
            print("acc averaged: ", acc_f_averaged)

            obs_len = 20
            pred_len = 30
            freq = 10

            dist_around_wo_acc = vel_f_averaged * (pred_len/freq)

            dist_around_acc = vel_f_averaged * (pred_len/freq) + 1/2 * acc_f_averaged_aux * (pred_len/freq)**2

            print("Estimated horizon without acceleration: ", dist_around_wo_acc)
            print("Estimated horizon with acceleration: ", dist_around_acc)

            if extended_xy_f.size > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(
                        xy_f[0, :],
                        xy_f[1, :],
                        ".",
                        color="b",
                        alpha=1,
                        linewidth=3,
                        zorder=1,
                    )

                for i in range(xy_f.shape[1]):
                    plt.text(xy_f[0,i],xy_f[1,i],i)

                x_min = min(xy_f[0,:])
                x_max = max(xy_f[0,:])
                y_min = min(xy_f[1,:])
                y_max = max(xy_f[1,:])

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

                ax2.plot(
                        extended_xy_f[0, :],
                        extended_xy_f[1, :],
                        ".",
                        color="r",
                        alpha=1,
                        linewidth=3,
                        zorder=1,
                    )

                for i in range(extended_xy_f.shape[1]):
                    plt.text(extended_xy_f[0,i],extended_xy_f[1,i],i)

                x_min = min(extended_xy_f[0,:])
                x_max = max(extended_xy_f[0,:])
                y_min = min(extended_xy_f[1,:])
                y_max = max(extended_xy_f[1,:])

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
            else:
                plt.plot(
                        xy_f[0, :],
                        xy_f[1, :],
                        ".",
                        color="b",
                        alpha=1,
                        linewidth=3,
                        zorder=1,
                    )

                for i in range(xy_f.shape[1]):
                    plt.text(xy_f[0,i],xy_f[1,i],i)

                x_min = min(xy_f[0,:])
                x_max = max(xy_f[0,:])
                y_min = min(xy_f[1,:])
                y_max = max(xy_f[1,:])

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

            plt.show()
            plt.close('all')

        xy_f = xy_f.T # TODO: Check the above xy_f and transform from 2 x 20 to 20 x 2 (easy to read)
        extended_xy_f = extended_xy_f.T

        return vel_f_averaged, acc_f_averaged, xy_f, extended_xy_f

    def interpolate_centerline(self,centerline,max_points=40,agent_xy=None,obs_len=None,seq_len=None,split=None,seq_id=None,viz=False):
        """
        """

        try:
            cx, cy = centerline[:,0], centerline[:,1]
            points = np.arange(cx.shape[0])

            new_points = np.linspace(points.min(), points.max(), max_points)
        
            new_cx = sp.interpolate.interp1d(points,cx,kind='cubic')(new_points)
            new_cy = sp.interpolate.interp1d(points,cy,kind='cubic')(new_points)
        except:
            pdb.set_trace()
            return

        interp_centerline = np.hstack([new_cx.reshape(-1,1),new_cy.reshape(-1,1)])

        if viz:
            fig, ax = plt.subplots(figsize=(8,8), facecolor="white")
            visualize_centerline(interp_centerline)

            if agent_xy.shape[0] == seq_len: # train and val
                # Agent observation

                plt.plot(
                    agent_xy[:obs_len, 0],
                    agent_xy[:obs_len, 1],
                    "-",
                    color="b",
                    alpha=1,
                    linewidth=3,
                    zorder=1,
                )

                # Agent prediction

                plt.plot(
                    agent_xy[obs_len:, 0],
                    agent_xy[obs_len:, 1],
                    "-",
                    color="r",
                    alpha=1,
                    linewidth=3,
                    zorder=1,
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
                    zorder=2,
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
                    zorder=1,
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
                    zorder=2,
                )

            output_dir = os.path.join(BASE_DIR,f"data/datasets/argoverse/motion-forecasting/{split}/map_features")

            if not os.path.exists(output_dir):
                print("Create output folder: ", output_dir)
                os.makedirs(output_dir) # makedirs creates intermediate folders

            filename = os.path.join(output_dir,f"{seq_id}_interpolated_oracle.png")

            plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

            plt.show()
            plt.close('all')

        return interp_centerline

    def get_yaw(self, agent_xy, obs_len):
        """
        Assuming the agent observation has been filtered, determine the agent's orientation
        in the last observation
        """

        lane_dir_vector = agent_xy[obs_len-1,:] - agent_xy[obs_len-2,:]
        yaw = math.atan2(lane_dir_vector[1],lane_dir_vector[0])
        pdb.set_trace()
        return lane_dir_vector, yaw

    def apply_tf(source_location, transform):  
        """
        """
        centroid = np.array([0.0,0.0,0.0,1.0]).reshape(4,1)

        centroid[0,0] = source_location[0]
        centroid[1,0] = source_location[1]
        centroid[2,0] = source_location[2]

        target_location = np.dot(transform,centroid) # == transform @ centroid

        return target_location 

    def rotz2D(self,yaw):
        """ 
        Rotation about the z-axis
        """
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c,  -s],
                        [s,    c]])

    def debug_centerline_and_agent(self, candidate_centerlines, xy, obs_len, seq_len, split, seq_id):
        """
        """

        fig, ax = plt.subplots(figsize=(8,8), facecolor="white")

        for centerline_coords in candidate_centerlines:
            # fig = visualize_centerline(centerline_coords)

            visualize_centerline(centerline_coords)

        if xy.shape[0] == seq_len: # train and val
            # Agent observation

            plt.plot(
                xy[:obs_len, 0],
                xy[:obs_len, 1],
                "-",
                color="b",
                alpha=1,
                linewidth=3,
                zorder=1,
            )

            # Agent prediction

            plt.plot(
                xy[obs_len:, 0],
                xy[obs_len:, 1],
                "-",
                color="r",
                alpha=1,
                linewidth=3,
                zorder=1,
            )

            # First observation

            first_obs_x = xy[0,0]
            first_obs_y = xy[0,1]

            plt.plot(
                first_obs_x,
                first_obs_y,
                "*",
                color="b",
                alpha=1,
                markersize=5,
                zorder=2,
            )

            # Last observation

            last_obs_x = xy[obs_len-1,0]
            last_obs_y = xy[obs_len-1,1]

            plt.plot(
                last_obs_x,
                last_obs_y,
                "o",
                color="b",
                alpha=1,
                markersize=5,
                zorder=2,
            )

            # Final GT

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            plt.plot(
                final_x,
                final_y,
                "o",
                color="r",
                alpha=1,
                markersize=5,
                zorder=2,
            )
        else: # test
            # Agent observation
            
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="b",
                alpha=1,
                linewidth=3,
                zorder=1,
            )

            # First observation

            first_obs_x = xy[0,0]
            first_obs_y = xy[0,1]

            plt.plot(
                first_obs_x,
                first_obs_y,
                "o",
                color="b",
                alpha=1,
                markersize=5,
                zorder=1,
            )

            # Last observation

            last_obs_x = xy[obs_len-1,0]
            last_obs_y = xy[obs_len-1,1]

            plt.plot(
                last_obs_x,
                last_obs_y,
                "*",
                color="b",
                alpha=1,
                markersize=5,
                zorder=2,
            )

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        # plt.axis("off")
        plt.title(f"Number of candidates = {len(candidate_centerlines)}")   

        output_dir = os.path.join(BASE_DIR,f"data/datasets/argoverse/motion-forecasting/{split}/map_features")
        if not os.path.exists(output_dir):
            print("Create output folder: ", output_dir)
            os.makedirs(output_dir) # makedirs creates intermediate folders

        filename = os.path.join(output_dir,f"{seq_id}_debug.png")

        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

    def get_closest_wp(self, pos, centerline):
        """
        """

        query_obs = np.repeat(pos.reshape(1,-1),repeats=centerline.shape[0],axis=0)
        dist_array = np.zeros((query_obs.shape[0]))

        for k in range(query_obs.shape[0]): # TODO: How to compute the L2-distance point by point?
            dist_array[k] = np.linalg.norm(centerline[k,:]-query_obs[k,:])

        closest_wp = np.argmin(dist_array)

        return closest_wp, dist_array

    def get_candidate_centerlines_for_trajectory(
            self,
            xy: np.ndarray,
            city_name: str,
            seq_id: int,
            avm: ArgoverseMap,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50,
            max_candidates: int = 6,
            mode: str = "test",
            split: str = "train",
            algorithm: str = "competition", # competition, map_api, get_around
            debug: bool = True
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold.
        Algorithm:
        1. Take the lanes in the bubble of last observed coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines based on point in polygon score.
        Args:
            xy: Trajectory coordinates, 
            city_name: City name, 
            avm: Argoverse map_api instance, 
            viz: Visualize candidate centerlines, 
            max_search_radius: Max search radius for finding nearby lanes in meters,
            seq_len: Sequence length, 
            max_candidates: Maximum number of centerlines to return, 
            mode: train/val/test mode
        Returns:
            candidate_centerlines: List of candidate centerlines
        """

        output_dir = os.path.join(BASE_DIR,f"data/datasets/argoverse/motion-forecasting/{split}/map_features")
        if not os.path.exists(output_dir):
            print("Create output folder: ", output_dir)
            os.makedirs(output_dir) # makedirs creates intermediate folders

        obs_len = 20
        pred_len = 30
        freq = 10

        min_dist_around = 15

        traj_len = xy.shape[0]
        agent_traj = copy.deepcopy(xy)
        
        if debug: # If debug, compute the centerlines with only the observation data 
                  # (even with train and val)
            xy = xy[:obs_len,:]

        # Filter agent's trajectory (smooth)

        vel, acc, xy_filtered, extended_xy_filtered = self.get_agent_velocity_and_acceleration(xy,
                                                                                               filter="least_squares",
                                                                                               debug=False)
                                                                                               
        dist_around = vel * (pred_len/freq) + 1/2 * acc * (pred_len/freq)**2
        
        if dist_around < min_dist_around:
            dist_around = min_dist_around

        ## Roughly compute the distance travelled with naive polynomial extension

        distance_travelled = 0
        max_dist = 100 # Hypothesis: max distance in 3 s

        index_max_dist = -1

        for i in range(extended_xy_filtered.shape[0]-1):
            if i >= obs_len:
                dist = np.linalg.norm((extended_xy_filtered[i+1,:] - extended_xy_filtered[i,:]))
                distance_travelled += dist

                if distance_travelled > max_dist and index_max_dist == -1:
                    index_max_dist = i

        reference_point = extended_xy_filtered[index_max_dist,:]
        # reference_point = extended_xy_filtered[obs_len-1,:]

        # Compute agent's orientation

        lane_dir_vector, yaw = self.get_yaw(xy_filtered, obs_len)

        if algorithm == "competition":
        # Compute centerlines (oracle or N) using Argoverse Forecasting baseline

            # Get all lane candidates within a bubble

            curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
                xy_filtered[-1, 0], xy_filtered[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

            # Keep expanding the bubble until at least 1 lane is found

            while (len(curr_lane_candidates) < 1
                and self._MANHATTAN_THRESHOLD < max_search_radius):
                self._MANHATTAN_THRESHOLD *= 2
                curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
                    xy_filtered[-1, 0], xy_filtered[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

            assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

            # Set dfs threshold

            # 10 represents the frequency in which the obstacles positions have been sampled:
            #   (steps / steps/s) -> s
            # self._DFS_THRESHOLD_FRONT_SCALE and self._DFS_THRESHOLD_FRONT_SCALE represent the velocities to
            # traverse frontwards and backwards in m/s:
            #   m/s · s = m

            # dfs_threshold_front = (self._DFS_THRESHOLD_FRONT_SCALE * (seq_len + 1 - traj_len) / 10)
            # dfs_threshold_back = self._DFS_THRESHOLD_BACK_SCALE * (traj_len + 1) / 10
            dfs_threshold_front = dfs_threshold_back = dist_around

            # print("dfs_threshold_front, dfs_threshold_back: ", dfs_threshold_front, dfs_threshold_back)

            # DFS to get all successor and predecessor candidates
            obs_pred_lanes: List[Sequence[int]] = []
            for lane in curr_lane_candidates:
                candidates_future = avm.dfs(lane, city_name, 0, dfs_threshold_front)
                candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back, True)

                # Merge past and future
                for past_lane_seq in candidates_past:
                    for future_lane_seq in candidates_future:
                        assert (
                            past_lane_seq[-1] == future_lane_seq[0]
                        ), "Incorrect DFS for candidate lanes past and future"
                        obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])
                        obs_pred_lanes.append(future_lane_seq)

                # # Only future
                # for future_lane_seq in candidates_future:
                #     obs_pred_lanes.append(future_lane_seq[1:])
                
            # Removing overlapping lanes
            obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

            # Sort lanes based on point in polygon score
            obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
                obs_pred_lanes, xy_filtered, city_name, avm)

            # If the best centerline is not along the direction of travel, re-sort
            if mode == "test":
                candidate_centerlines = self.get_heuristic_centerlines_for_test_set(
                    obs_pred_lanes, xy_filtered, city_name, avm, max_candidates, scores)
            else:
                candidate_centerlines = avm.get_cl_from_lane_seq(
                    [obs_pred_lanes[0]], city_name)
                
            ## Additional ##

            # Sort centerlines based on the distance to a reference point, usually the last observation

            distances = []
            for centerline in candidate_centerlines:
                distances.append(min(np.linalg.norm((centerline - reference_point),axis=1)))
            
            # If we want to filter those lanes with the same distance to reference point
            # TODO: Is this hypothesis correct?
            
            # unique_distances = list(set(distances))
            # unique_distances.sort()
            # unique_distances = unique_distances[:max_candidates]

            # # print("unique distances: ", unique_distances)
            # final_indeces = [np.where(distances == unique_distance)[0][0] for unique_distance in unique_distances]

            # final_candidates = []
            # for index in final_indeces:
            #     final_candidates.append(candidate_centerlines[index])

            sorted_indeces = np.argsort(distances)
            final_candidates = []
            for index in sorted_indeces:
                final_candidates.append(candidate_centerlines[index])
                
            candidate_centerlines = final_candidates

        elif algorithm == "map_api": 
        # Compute centerlines using Argoverse Map API

            # Get all lane candidates within a bubble
            manhattan_threshold = 5.0
            curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(xy_filtered[-1, 0], xy_filtered[-1, 1], city_name, manhattan_threshold)

            # Keep expanding the bubble until at least 1 lane is found
            while len(curr_lane_candidates) < 1 and manhattan_threshold < max_search_radius:
                manhattan_threshold *= 2
                curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(xy_filtered[-1, 0], xy_filtered[-1, 1], city_name, manhattan_threshold)

            assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

            # displacement = np.sqrt((xy[0, 0] - xy[obs_len-1, 0]) ** 2 + (xy[0, 1] - xy[obs_len-1, 1]) ** 2)
            # dfs_threshold = displacement * 2.0
            # dfs_threshold_front = dfs_threshold_back = dfs_threshold
                
            dfs_threshold_front = dist_around
            dfs_threshold_back = dist_around

            # DFS to get all successor and predecessor candidates
            obs_pred_lanes: List[List[int]] = []
            for lane in curr_lane_candidates:
                candidates_future = avm.dfs(lane, city_name, 0, dfs_threshold_front)
                candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back, True)

                # Merge past and future
                for past_lane_seq in candidates_past:
                    for future_lane_seq in candidates_future:
                        assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                        obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

            # Removing overlapping lanes
            obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

            # Remove unnecessary extended predecessors
            obs_pred_lanes = avm.remove_extended_predecessors(obs_pred_lanes, xy_filtered, city_name)

            # Getting candidate centerlines
            candidate_cl = avm.get_cl_from_lane_seq(obs_pred_lanes, city_name)

            # Reduce the number of candidates based on distance travelled along the centerline
            candidate_centerlines = filter_candidate_centerlines(xy_filtered, candidate_cl)

            # If no candidate found using above criteria, take the onces along with travel is the maximum
            if len(candidate_centerlines) < 1:
                candidate_centerlines = get_centerlines_most_aligned_with_trajectory(xy_filtered, candidate_cl)

            ## Additional ##

            # Sort centerlines based on the distance to a reference point, usually the last observation

            distances = []
            for centerline in candidate_centerlines:
                distances.append(min(np.linalg.norm((centerline - reference_point),axis=1)))
            
            unique_distances = list(set(distances))
            unique_distances.sort()
            unique_distances = unique_distances[:max_candidates]

            # print("unique distances: ", unique_distances)
            final_indeces = [np.where(distances == unique_distance)[0][0] for unique_distance in unique_distances]

            final_candidates = []
            for index in final_indeces:
                final_candidates.append(candidate_centerlines[index])

            # sorted_indeces = np.argsort(distances)
            # sorted_indeces = sorted_indeces[:max_candidates]
            # final_candidates = []
            # for index in sorted_indeces:
            #     final_candidates.append(candidate_centerlines[index])

            candidate_centerlines = final_candidates

        elif algorithm == "get_around":
        # Compute the lanes around a given position given a specific distance

            dist_rasterized_map = [-dist_around,dist_around,-dist_around,dist_around]

            xcenter, ycenter = xy_filtered[obs_len-1, 0], xy_filtered[obs_len-1, 1]
            x_min = xcenter + dist_rasterized_map[0]
            x_max = xcenter + dist_rasterized_map[1]
            y_min = ycenter + dist_rasterized_map[2]
            y_max = ycenter + dist_rasterized_map[3]

            seq_lane_props = dict()

            seq_lane_props = avm.city_lane_centerlines_dict[city_name]

            ### Get lane centerlines which lie within the range of trajectories

            lane_centerlines = []
            lane_ids = []
            obs_pred_lanes: List[Sequence[int]] = []

            yaw_aux = math.pi/2 - yaw # In order to align the data with the vertical axis
            R = self.rotz2D(yaw_aux)

            last_obs = xy_filtered[-1,:]

            for lane_id, lane_props in seq_lane_props.items():
                lane_cl = lane_props.centerline

                if (np.min(lane_cl[:, 0]) < x_max
                    and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min):

                    lane_centerlines.append(lane_cl)
                    lane_ids.append(lane_id)
            candidate_centerlines = lane_centerlines

            ### Filter centerline which start point is in front of the agent last observation

            # yaw_aux = math.pi/2 - yaw # In order to align the data with the vertical axis
            # R = map_features_utils_instance.rotz2D(yaw)
            # start_oracle = oracle_centerline[0,:]

            # oracle_centerline_aux = []
            # oracle_centerline_aux.append(np.dot(R,oracle_centerline.T).T)
            # agent_xy_aux = np.dot(R,agent_xy.T).T

            # map_features_utils_instance.debug_centerline_and_agent(oracle_centerline_aux, agent_xy_aux, obs_len, obs_len+pred_len, split_name, file_id)

        if viz:
            fig, ax = plt.subplots(figsize=(6,6), facecolor="black")

            xmin = ymin = 50000
            xmax = ymax = -50000

            # Paint centerlines

            if len(candidate_centerlines) > 0:
                for centerline_index, centerline_coords in enumerate(candidate_centerlines):
                    visualize_centerline(centerline_coords) # Uncomment this to check the start and end

                    if np.min(centerline_coords[:,0]) < xmin: xmin = np.min(centerline_coords[:,0])
                    if np.min(centerline_coords[:,1]) < ymin: ymin = np.min(centerline_coords[:,1])
                    if np.max(centerline_coords[:,0]) > xmax: xmax = np.max(centerline_coords[:,0])
                    if np.max(centerline_coords[:,1]) > ymax: ymax = np.max(centerline_coords[:,1])

                    lane_polygon = centerline_to_polygon(centerline_coords)

                                                                            #"black"  
                    ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "white", edgecolor='white', fill=True)

                    if centerline_index == 0:
                        ax.plot(centerline_coords[:, 0], centerline_coords[:, 1], "-", color="cyan", linewidth=3, alpha=1.0, zorder=2)
                    elif centerline_index < max_candidates:
                        ax.plot(centerline_coords[:, 0], centerline_coords[:, 1], "-", color="green", linewidth=3, alpha=1.0, zorder=2)
                    else:
                        ax.plot(centerline_coords[:, 0], centerline_coords[:, 1], "-", color="gray", linewidth=1.5, alpha=1.0, zorder=2)

                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.axis("off")

            ## Paint agent's orientation

            # dx = lane_dir_vector[0] * 10
            # dy = lane_dir_vector[1] * 10

            # plt.arrow(
            #     xy_filtered[obs_len-1,0], # Arrow origin
            #     xy_filtered[obs_len-1,1],
            #     dx, # Length of the arrow
            #     dy,
            #     color="darkmagenta",
            #     width=1.0,
            #     zorder=16,
            # )

            # if algorithm == "map_api":
            #     filename = os.path.join(output_dir,f"{seq_id}_binary_plausible_area.png")
            #     plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)

            if agent_traj.shape[0] == seq_len: # train and val
                # Agent observation

                plt.plot(
                    agent_traj[:obs_len, 0],
                    agent_traj[:obs_len, 1],
                    "-",
                    color="b",
                    alpha=1,
                    linewidth=3,
                    zorder=2,
                )

                # Agent prediction (GT)

                plt.plot(
                    agent_traj[obs_len:, 0],
                    agent_traj[obs_len:, 1],
                    "-",
                    color="r",
                    alpha=1,
                    linewidth=3,
                    zorder=2,
                )

                final_x = agent_traj[-1, 0]
                final_y = agent_traj[-1, 1]

                # Final position

                plt.plot(
                    final_x,
                    final_y,
                    "o",
                    color="r",
                    alpha=1,
                    markersize=5,
                    zorder=3,
                )

                ## Extended Agent's trajectory filtered

                if extended_xy_filtered.size > 0:
                    plt.plot(
                        # Since extended_xy_filtered has the input observation and then the extension
                        # assuming a polynomial function of order = 2. Here we only want to plot the 
                        # extension to roughly know where the agent would finish
                        extended_xy_filtered[obs_len:, 0],
                        extended_xy_filtered[obs_len:, 1],
                        "-",
                        color="purple",
                        alpha=1,
                        linewidth=3,
                        zorder=2,
                    )
            else: # test
                # Agent observation
                
                plt.plot(
                    agent_traj[:, 0],
                    agent_traj[:, 1],
                    "-",
                    color="b",
                    alpha=1,
                    linewidth=3,
                    zorder=2,
                )

                final_x = agent_traj[-1, 0]
                final_y = agent_traj[-1, 1]

                # Final position

                plt.plot(
                    final_x,
                    final_y,
                    "o",
                    color="b",
                    alpha=1,
                    markersize=5,
                    zorder=3,
                )

                ## Extended Agent's trajectory filtered

                if extended_xy_filtered.size > 0:
                    plt.plot(
                        # Since extended_xy_filtered has the input observation and then the extension
                        # assuming a polynomial function of order = 2. Here we only want to plot the 
                        # extension to roughly know where the agent would finish
                        extended_xy_filtered[obs_len:, 0],
                        extended_xy_filtered[obs_len:, 1],
                        "-",
                        color="purple",
                        alpha=1,
                        linewidth=3,
                        zorder=2,
                    )

            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("on")

            fig.patch.set_facecolor('white')
            plt.title(f"Number of candidates = {len(candidate_centerlines)}")   

            if mode == "test":
                filename = os.path.join(output_dir,f"{seq_id}_{algorithm}_relevant_centerlines.png")
            else:
                filename = os.path.join(output_dir,f"{seq_id}_{algorithm}_oracle.png")
            # print("filename: ", filename)
            plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)

            plt.show()
            plt.close('all')

        return candidate_centerlines

    def compute_map_features(
            self,
            agent_track: np.ndarray,
            seq_id: int,
            split: str,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
            avm: ArgoverseMap,
            viz: bool = False,
            max_candidates: int = 6,
            algorithm: str = "competition" # competition, map_api, get_around
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute map based features for the given sequence.
        If the mode is test, oracle_nt_dist will be empty, candidate_nt_dist will be populated.
        If the mode is train/val, oracle_nt_dist will be populated, candidate_nt_dist will be empty.
        Args:
            agent_track : Data for the agent track
            obs_len : Length of observed trajectory
            seq_len : Length of the sequence
            raw_data_format : Format of the sequence
            mode: train/val/test mode
            
        Returns:
            oracle_nt_dist (numpy array): normal and tangential distances for oracle centerline
                map_feature_helpers (dict): Dictionary containing helpers for map features
        """

        if max_candidates != self._MAX_CENTERLINE_CANDIDATES_TEST:
            self._MAX_CENTERLINE_CANDIDATES_TEST = max_candidates

        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        # Get candidate centerlines using observed trajectory
        if mode == "test":
            oracle_centerline = np.full((seq_len, 2), None)
            oracle_nt_dist = np.full((seq_len, 2), None)
            candidate_centerlines = self.get_candidate_centerlines_for_trajectory(
                # agent_xy_obs, # only observation
                agent_xy, # whole observation (if train/val, plot GT, even with multiple centerlines)
                city_name,
                seq_id,
                avm,
                viz=viz,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
                split=split,
                algorithm=algorithm
            )

            # Get nt distance for the entire trajectory using candidate centerlines
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)

        else:
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy, # whole trajectory
                city_name,
                seq_id,
                avm,
                viz=viz,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode=mode,
                split=split,
                algorithm=algorithm
            )[0]
            candidate_centerlines = [np.full((seq_len, 2), None)]
            candidate_nt_distances = [np.full((seq_len, 2), None)]

            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=viz)

        map_feature_helpers = {
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
        }

        return oracle_nt_dist, map_feature_helpers

