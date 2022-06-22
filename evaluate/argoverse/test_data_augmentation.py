# Rotation (If the image is rotated, all trajectories must be rotated)

        # rotate_seq = 0

        # if rotate_seq:
        #     curr_obj_seq = dataset_utils.rotate_traj(curr_obj_seq,rotation_angle)

        # data_aug_flag = 0

        # if split == "train" and (data_aug_flag == 1 or augs):
        #     # Add data augmentation

        #     if not augs:
        #         print("Get comb")
        #         augs = dataset_utils.get_data_aug_combinations(3) # Available data augs: Swapping, Erasing, Gaussian noise

        #     ## 1. Swapping

        #     if augs[0]:
        #         print("Swapping")
        #         curr_obj_seq = dataset_utils.swap_points(curr_obj_seq,num_obs=obs_len)

        #     ## 2. Erasing

        #     if augs[1]:
        #         print("Erasing")
        #         curr_obj_seq = dataset_utils.erase_points(curr_obj_seq,num_obs=obs_len,percentage=0.3)

        #     ## 3. Add Gaussian noise

        #     if augs[2]:
        #         print("Gaussian")
        #         curr_obj_seq = dataset_utils.add_gaussian_noise(curr_obj_seq,num_obs=obs_len,mu=0,sigma=0.5)