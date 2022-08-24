# Example Usage:
# python sg.py position.dat 7 2

import math
import sys
import git
import os
import pdb

import numpy as np
import numpy.linalg
import pylab as py
import csaps

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import model.datasets.argoverse.dataset as dataset
import model.datasets.argoverse.dataset_utils as dataset_utils

if __name__ == '__main__':
    dataset_path = "data/datasets/argoverse/motion-forecasting/"
    split = "train"
    data_folder = BASE_DIR + "/" + dataset_path + split + "/data/"

    files, num_files = dataset_utils.load_list_from_folder(data_folder)

    file_id_list = []
    root_file_name = None
    for file_name in files:
        if not root_file_name:
            root_file_name = os.path.dirname(os.path.abspath(file_name))
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        file_id_list.append(file_id)
    file_id_list.sort()

    file_id = "50008"
    path = os.path.join(root_file_name,str(file_id)+".csv")
    data = dataset_utils.read_file(path) 

    frames = np.unique(data[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

    idx = 0

    obs_len = 20
    pred_len = 30
    obs_origin = 20

    num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
    curr_seq_rel, id_frame_list, object_class_list, city_id, map_origin = \
        dataset.process_window_sequence(idx, frame_data, frames, \
                                        obs_len, pred_len, file_id, split, obs_origin)

    # 2. Get AGENT information

    agent_index = np.where(object_class_list == 1)[0].item()
    agent_seq = curr_seq[agent_index,:,:].transpose() # Whole sequence

    obs_data = agent_seq[:obs_len,:]
    
    import numpy as np
    from csaps import csaps
    import matplotlib.pyplot as plt

    x = obs_data[:,0]
    y = obs_data[:,1]
    pdb.set_trace()
    t = np.linspace(0.1, 2, 20)

    # data_i = csaps(x, y, t)
    data_i = csaps(obs_data,t,smooth=0.95)
    xi = data_i[0, :]
    yi = data_i[1, :]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(x, y, ':o', xi, yi, '*')
    plt.savefig("csaps.png", bbox_inches='tight', facecolor=fig.get_facecolor(), 
                        edgecolor='none', pad_inches=0)



https://github.com/espdev/sgolay2/issues/3