import git
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.dataset as dataset
import model.datasets.argoverse.dataset_utils as dataset_utils 
import model.datasets.argoverse.map_functions as map_functions
import model.datasets.argoverse.goal_points_functions as goal_points_functions

from model.datasets.argoverse.map_functions import MapFeaturesUtils
map_features_utils_instance = MapFeaturesUtils()

def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

# Set root_dir to the correct path to your dataset folder

split_name = "train"
split_percentage = 1.0
root_dir = os.path.join(BASE_DIR,
                        f'data/datasets/argoverse/motion-forecasting/{split_name}')
data_processed_dir = os.path.join(root_dir, f'data_processed_{str(int(split_percentage*100))}_percent')

# Read all plausible areas and store in a single npy array

subfolder = "map_features"
files, num_files = dataset_utils.load_list_from_folder(os.path.join(root_dir,subfolder))

file_id_list = []
root_file_name = None
for file_name in files:
    if not root_file_name:
        root_file_name = os.path.dirname(os.path.abspath(file_name))
    if "npy" in file_name or "npz" in file_name:
        pass
    else:
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('_')[0])
        file_id_list.append(file_id)
file_id_list.sort()
print("Num files: ", num_files)

mask = 255
rows = 400 # Default number of rows

save_fig = False
show = True
change_img_bg = False

relevant_centerlines_filtered_npz = np.load(os.path.join(data_processed_dir,"relevant_centerlines.npz"),allow_pickle=True)
relevant_centerlines_filtered = relevant_centerlines_filtered_npz['arr_0'].item()

plausible_area_indeces_list = []

check_every = 0.1 # (· 100) % of total files
check_every_n_files = int(len(file_id_list)*check_every)
print(f"Check remaining time every {check_every_n_files} files")
time_per_iteration = float(0)
aux_time = float(0)

file_id_list = file_id_list[150000:]

for i,file_id in enumerate(file_id_list):
    # print(">>>>>> File id: ", file_id)

    start_t = time.time()

    files_remaining = len(file_id_list) - (i+1)

    relevant_centerlines = relevant_centerlines_filtered[str(file_id)]

    real_world_width = relevant_centerlines["real_world_width"]
    real_world_height = relevant_centerlines["real_world_height"]
    # print("RW width, height: ", real_world_width, real_world_height)

    cols = math.ceil(rows*(real_world_width/real_world_height)) 
    scale_x = float(cols/real_world_width) # px/m
    scale_y = float(rows/real_world_height) # px/m
    center_px = (int(rows/2),int(cols/2))

    filename = os.path.join(root_file_name,f"{file_id}_binary_plausible_area_filtered.png")
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # print("Orig shape: ", img_gray.shape)
    img_gray = cv2.resize(img_gray, dsize=(cols,rows))

    discretized_area_points = 512

    indeces_white = np.vstack(np.where(img_gray == 255)).T
    # print("Ind white: ", indeces_white.shape)

    try:
        subsampling = int(math.floor(indeces_white.shape[0] / discretized_area_points))

        indeces_white_sub = indeces_white[::subsampling]
        # print("Ind white sub: ", indeces_white_sub.shape)
        fe_x = indeces_white_sub[:,1]
        fe_y = indeces_white_sub[:,0]
    except:
        pdb.set_trace()

    # start = time.time()

    # subsampling_rows = int(rows / 100)
    # subsampling_cols = int(cols / 100)
    # print("Subsampling rows, columns: ", subsampling_rows, subsampling_cols)
    # # print("Subsampling: ", subsampling)

    # total_indeces_rows = np.arange(rows)
    # total_indeces_cols = np.arange(cols)

    # print("pre rows, pre columns: ", rows, cols)

    # # while(len(fe_x) < discretized_area_points):
    # rows_subsampled = total_indeces_rows[::subsampling_rows]
    # cols_subsampled = total_indeces_cols[::subsampling_cols]
    # print("Rows, cols: ", len(rows_subsampled), len(cols_subsampled))

    # start1 = time.time()
    # index_pair = []
    # for r in rows_subsampled:
    #     for c in cols_subsampled:
    #         pair = np.array([r,c])
    #         index_pair.append(pair)
    # index_pair = np.array(index_pair)
    # end1 = time.time()
    # # print(">> Pairs subsampled: ", end1-start1)

    # start2 = time.time()
    # arr = intersect2D(index_pair,indeces_white)
    # end2 = time.time()
    # # print(">> Intersect time: ", end2-start2)

    # fe_x = arr[:,1]
    # fe_y = arr[:,0]

    # end = time.time()  
    # # print("Time: ", end-start)

    # # sample_index = np.random.randint(low=0,high=len(fe_x),size=discretized_area_points)
    # # sample_index = np.sort(sample_index) # Sort indeces in order to have to pixels sorted (from top-left to bottom right)
    # # fe_x, fe_y = fe_x[sample_index], fe_y[sample_index]

    # if len(fe_x) < discretized_area_points:
    #     subsampling = int(subsampling / 2)

    if len(fe_x) < discretized_area_points:
        print(">>>>>> File id: ", file_id)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB) 
        save_fig = True
        show = False
        filename2 = os.path.join(root_file_name,f"{file_id}_binary_plausible_area_filtered_discrete_bug.png")
        goal_points_functions.plot_fepoints(img, filename2, goals_px_x=fe_x, goals_px_y=fe_y,
                                        save_fig=save_fig, show=show, change_bg=change_img_bg)
        pdb.set_trace()
    plausible_area_indeces = np.vstack((fe_y,fe_x)).T
    plausible_area_indeces_list.append(plausible_area_indeces)

    end_t = time.time()

    # Data aug

    # limit = 4
    # noise = np.random.randint(-limit,limit,(len(fe_x),2))
    # fe_x = fe_x + noise[:,1]
    # fe_y = fe_y + noise[:,0]

    # fe_x = indeces[1][::subsampling]
    # fe_y = indeces[0][::subsampling]

    aux_time += (end_t-start_t)
    time_per_iteration = aux_time/(i+1)

    if i % check_every_n_files == 0:
        print(f"Time per iteration: {time_per_iteration} s. \n \
                Estimated time to finish ({files_remaining} files): {round(time_per_iteration*files_remaining/60)} min")

plausible_area_indeces_array = np.array(plausible_area_indeces_list, dtype="object")

filename_array = os.path.join(root_dir,"map_features_indeces.npz")
with open(filename_array, 'wb') as my_file: np.savez_compressed(my_file, plausible_area_indeces_array)