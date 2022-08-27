import cv2
import numpy as np
import os
import copy
import pdb
import sys
import math
import git
import csv
import torch
import matplotlib.pyplot as plt
import time

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.dataset as dataset
import model.datasets.argoverse.dataset_utils as dataset_utils 
import model.datasets.argoverse.goal_points_functions as goal_points_functions                                         

# Load files

dataset_path = "data/datasets/argoverse/motion-forecasting/"
split = "train"
split_folder = BASE_DIR + "/" + dataset_path + split
data_folder = BASE_DIR + "/" + dataset_path + split + "/data/"
data_images_folder = BASE_DIR + "/" + dataset_path + split + "/data_images"
# data_images_folder = BASE_DIR + "/" + dataset_path + split + "/data_images_150mx150m"
goal_points_folder = split_folder + "/goal_points_test"

files, num_files = dataset_utils.load_list_from_folder(data_folder)

file_id_list = []
root_file_name = None
for file_name in files:
    if not root_file_name:
        root_file_name = os.path.dirname(os.path.abspath(file_name))
    file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
    file_id_list.append(file_id)
file_id_list.sort()

limit = 1000
if limit != -1:
    file_id_list = file_id_list[:limit]

print("Num files: ", len(file_id_list))

# Iterate over the whole split -> Compute the K-most plausible goals -> Compare to groundtruth end-point

obs_origin = 20
obs_len = 20
pred_len = 30
seq_len = obs_len + pred_len

show_pred_gt = False
show = False
save_fig = False
change_img_bg = False
num_initial_samples = 1000
pred_seconds = 3 # [seconds] To compute the goal points 
                 #           until pred_seconds ahead (assuming constant velocity)
NUM_GOAL_POINTS = 32
MAX_CLUSTERS = 6 # Maximum number of modes (multimodality)
MULTIMODAL = True 

dist_around = 40
real_world_offset = dist_around

aux_list = []
closest_gp_dist_list = []

for t,file_id in enumerate(file_id_list):
    print(f"\nFile {file_id} -> {t+1}/{len(file_id_list)}")

    path = os.path.join(root_file_name,str(file_id)+".csv")
    data = dataset_utils.read_file(path) 
    origin_pos, city_name = dataset_utils.get_origin_and_city(data,obs_origin)

    # 0. Load image (physical information)

    filename = data_images_folder + "/" + str(file_id) + ".png"

    img = cv2.imread(filename)
    img = cv2.resize(img, dsize=(600,600))
    height, width = img.shape[:2]
    img_size = height
    scale_x = scale_y = float(height/(2*real_world_offset))

    cx = int(width/2)
    cy = int(height/2)
    car_px = (cy,cx)

    # 1. Get past observations

    frames = np.unique(data[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

    idx = 0

    num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
    curr_seq_rel, id_frame_list, object_class_list, city_id, map_origin = \
        dataset.process_window_sequence(idx, frame_data, frames, \
                                        obs_len, pred_len, file_id, split, obs_origin)

    # 2. Get AGENT information

    agent_index = np.where(object_class_list == 1)[0].item()
    agent_seq = curr_seq[agent_index,:,:].transpose() # Whole sequence

    agent_seq_global = agent_seq + origin_pos # abs (hdmap coordinates)
    agent_seq_global = torch.from_numpy(agent_seq_global)
    agent_obs_seq_global = agent_seq_global[:obs_origin,:]

    # px = pixels
    agent_px_points = goal_points_functions.transform_real_world2px(agent_seq_global, origin_pos, real_world_offset, img_size)
    agent_px_x, agent_px_y = agent_px_points[:,0], agent_px_points[:,1] # TODO: Refactorize

    # 3. Get feasible area points (N samples)

    # 3.0. (Optional) Observe random sampling in the whole feasible area

    rad = 1000 # meters. Cause we want to observe all points around the AGENT

    fe_y, fe_x = goal_points_functions.get_points(img, car_px, scale_x, rad=10000, color=255, N=num_initial_samples, 
                            sample_car=True, max_samples=None) # return rows, columns

    # 3.1. Determine AGENT's acceleration

    # goal_points_functions.get_agent_acceleration(torch.transpose(agent_obs_seq_global,0,1))

    # 3.1. Filter using AGENT estimated velocity in the last observation frame

    mean_vel = goal_points_functions.get_agent_velocity(torch.transpose(agent_obs_seq_global,0,1))
    radius = mean_vel * pred_seconds
    radius_px = radius * scale_x	

    fe_y, fe_x = goal_points_functions.get_points(img, car_px, scale_x, rad=radius_px, color=255, N=1024, 
                                sample_car=True, max_samples=None) # return rows, columns

    # 3.2. Filter using AGENT estimated orientation (yaw) in the last observation frame

    mean_yaw = goal_points_functions.get_agent_yaw(torch.transpose(agent_obs_seq_global,0,1)) # radians

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

    # 4. Get furthest N samples (closest the the hypothetical radius)

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

    # 5. Clustering

    final_samples = np.hstack((final_samples_x.reshape(-1,1),final_samples_y.reshape(-1,1)))
    final_samples.shape

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import numpy as np

    # Initialize the class object

    kmeans = KMeans(n_clusters = MAX_CLUSTERS)
    
    # Predict the labels of clusters

    label = kmeans.fit_predict(final_samples)

    # Getting unique labels
    
    u_labels = np.unique(label)

    # Final clusters

    aux = torch.zeros((MAX_CLUSTERS,2))
    cont_goals = torch.zeros((MAX_CLUSTERS))

    for i in range(final_samples.shape[0]):
        aux[label[i],0] += final_samples[i,0]
        aux[label[i],1] += final_samples[i,1]
        cont_goals[label[i]] += 1 # Num goals per cluster
    aux = torch.div(aux,cont_goals.reshape(MAX_CLUSTERS,1))
    label = torch.arange(MAX_CLUSTERS)

    # 6. Transform pixels to real-world coordinates

    final_samples_px = np.hstack((aux[:,1].reshape(-1,1), aux[:,0].reshape(-1,1))) # rows (y), columns (x)
    rw_points = goal_points_functions.transform_px2real_world(final_samples_px, origin_pos, real_world_offset, img_size)

    # 7. Check error with gt end point (AGENT)

    end_point_gt_map = agent_seq_global[-1,:]

    closest_gp_dist = 50000

    for rw_point in rw_points:
        try:
            dist = np.linalg.norm(rw_point - end_point_gt_map)
        except:
            dist = np.linalg.norm(rw_point - end_point_gt_map.cpu().detach().numpy())

        if dist < closest_gp_dist:
            closest_gp_dist = dist

    _aux_list = [t,file_id,closest_gp_dist]
    aux_list.append(_aux_list)
    closest_gp_dist_list.append(closest_gp_dist)

    print("Error: ", closest_gp_dist)

# 8. Store in .csv

path = os.path.join(BASE_DIR,"evaluate/argoverse",split+"_goal_points_error.csv")

with open(path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    header = ['Index','Sequence.csv','Closest_GP_dist']
    csv_writer.writerow(header)

    for __aux_list in aux_list:
        csv_writer.writerow(__aux_list)

    closest_gp_dist_array = np.array(closest_gp_dist_list)

    mean = np.mean(closest_gp_dist_array)
    median = np.median(closest_gp_dist_array)

    # Write mean and median

    csv_writer.writerow(['-','-','-'])
    csv_writer.writerow(['-','-','Mean',mean])
    csv_writer.writerow(['-','-','Median',median])
    


