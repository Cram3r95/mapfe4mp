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
from typing import Any, Dict, List, Tuple, Union
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

# DL & Math imports

import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
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
                  dist_rasterized_map,
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

# Compute map features around a specific obstacle position 

_MANHATTAN_THRESHOLD = 5.0 # 5.0  # meters
_DFS_THRESHOLD_FRONT_SCALE = 20.0  # meters
_DFS_THRESHOLD_BACK_SCALE = 20.0  # meters
_MAX_SEARCH_RADIUS_CENTERLINES = 10.0  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 8 # 10

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
        #lane_seq_polygon = cascaded_union([
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

    def get_candidate_centerlines_for_trajectory(
            self,
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50,
            max_candidates: int = 10,
            mode: str = "test",
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
        # Get all lane candidates within a bubble
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
            xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        # Keep expanding the bubble until at least 1 lane is found
        while (len(curr_lane_candidates) < 1
               and self._MANHATTAN_THRESHOLD < max_search_radius):
            self._MANHATTAN_THRESHOLD *= 2
            curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
                xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # Set dfs threshold
        traj_len = xy.shape[0]

        # Assuming a speed of 50 mps, set threshold for traversing in the front and back # TODO: ??
        # dfs_threshold_front = (self._DFS_THRESHOLD_FRONT_SCALE *
        #                        (seq_len + 1 - traj_len) / 10)
        # dfs_threshold_back = self._DFS_THRESHOLD_BACK_SCALE * (traj_len +
        #                                                        1) / 10
        dfs_threshold_front = 50
        dfs_threshold_back = 1
 
        dist2stopsearch_front = 0
        dist2stopsearch_back = 2 # I do not know the exact working principle, but in order to avoid calculating
                                 # centerlines which end is behind the vehicle, dfs_threshold_back and dist2stopsearch_back
                                 # must be different from 0 and dfs_threshold_back < dist2stopsearch_back


        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[Sequence[int]] = []
        for index,lane in enumerate(curr_lane_candidates):
            if index>0:
                break
            candidates_future = avm.dfs(lane, city_name, dist2stopsearch_front, dfs_threshold_front)
            candidates_past = avm.dfs(lane, city_name, dist2stopsearch_back, dfs_threshold_back,True)

            # Merge past and future
            # for past_lane_seq in candidates_past:
            #     for future_lane_seq in candidates_future:
            #         assert (
            #             past_lane_seq[-1] == future_lane_seq[0]
            #         ), "Incorrect DFS for candidate lanes past and future"
                    
            #         obs_pred_lanes.append(past_lane_seq)# + future_lane_seq[1:])
                    # obs_pred_lanes.append(future_lane_seq[1:])

            for future_lane_seq in candidates_future:
                obs_pred_lanes.append(future_lane_seq[1:])
        #pdb.set_trace()
        # obs_pred_lanes = [curr_lane_candidates]
        end = time.time()



        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Sort lanes based on point in polygon score
        obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
            obs_pred_lanes, xy, city_name, avm)

        # If the best centerline is not along the direction of travel, re-sort
        if mode == "test":
            candidate_centerlines = self.get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy, city_name, avm, max_candidates, scores)
        else:
            candidate_centerlines = avm.get_cl_from_lane_seq(
                [obs_pred_lanes[0]], city_name)

        #########################
        # TESTING

        dist_rasterized_map = [-40,40,-40,40]

        xcenter, ycenter = xy[-1, 0], xy[-1, 1]
        x_min = xcenter + dist_rasterized_map[0]
        x_max = xcenter + dist_rasterized_map[1]
        y_min = ycenter + dist_rasterized_map[2]
        y_max = ycenter + dist_rasterized_map[3]

        seq_lane_props = dict()

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
        candidate_centerlines = lane_centerlines
        #########################





        if viz:
            fig = plt.figure(0, figsize=(8, 8))
            for centerline_coords in candidate_centerlines:
                # fig = visualize_centerline(centerline_coords)
                visualize_centerline(centerline_coords)

            # Agent trajectory

            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=3,
                zorder=15,
            )

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            # Final position

            plt.plot(
                final_x,
                final_y,
                "o",
                color="#d33e4c",
                alpha=1,
                markersize=10,
                zorder=15,
            )
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title(f"Number of candidates = {len(candidate_centerlines)}")
            #plt.show()
            
            filename = "/home/denso/carlos_vsr_workspace/mapfe4mp/evaluate/argoverse/test_centerlines.png"
            plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), 
                edgecolor='none', pad_inches=0)
            #pdb.set_trace()
        return candidate_centerlines

    def compute_map_features(
            self,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
            avm = ArgoverseMap,
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

        start1 = time.time()
        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        print("Time 1: ", time.time()-start1)
        # Get candidate centerlines using observed trajectory
        
        if mode == "test":
            start2 = time.time()
            oracle_centerline = np.full((seq_len, 2), None)
            oracle_nt_dist = np.full((seq_len, 2), None)
            candidate_centerlines = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=True,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
            )
            print("Time 2: ", time.time()-start2)
            # Get nt distance for the entire trajectory using candidate centerlines
            start3 = time.time()
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)
            print("Time 3: ", time.time()-start3)
        else:
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=True,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode=mode,
            )[0]
            candidate_centerlines = [np.full((seq_len, 2), None)]
            candidate_nt_distances = [np.full((seq_len, 2), None)]

            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=False)

        map_feature_helpers = {
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
        }

        return oracle_nt_dist, map_feature_helpers