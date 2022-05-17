import yaml
import json
import logging
import os
import sys
import pdb
import argparse

from datetime import datetime
from prodict import Prodict
from pathlib import Path

TRAINER_LIST = ["so", "so2", "sovi", "trans_so", "trans_sovi", "soconf", "settrans",
                "settransgoal", "so_goals", "so_data_augs", "so_goals_gan", 
                "soconf_goals", "soconf_goals_cgh"]

def create_logger(file_path):
    FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    # formatter = logging.Formatter('[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s')

    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stdout_handler.setLevel(logging.DEBUG)
    # stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stdout_handler)
    return logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=str, choices=TRAINER_LIST)
    args = parser.parse_args()
    print(args.trainer)
    
    trainer = None
    if args.trainer == "so":
        from sophie.trainers.trainer_gen_so import model_trainer
        config_path = "./configs/mp_so.yml"
    elif args.trainer == "so_goals":
        from sophie.trainers.trainer_gen_so_goals import model_trainer
        config_path = "./configs/mp_so_goals.yml"
    elif args.trainer == "so_goals_gan":
        from sophie.trainers.trainer_gen_so_goals_gan import model_trainer
        config_path = "./configs/mp_so_goals_gan.yml"
    elif args.trainer == "so_data_augs":
        from sophie.trainers.trainer_gen_so_data_augs import model_trainer
        config_path = "./configs/mp_so_data_augs.yml"
    elif args.trainer == "soconf":
        from sophie.trainers.trainer_gen_soconf import model_trainer
        config_path = "./configs/mp_soconf.yml"
    elif args.trainer == "sovi":
        from sophie.trainers.trainer_gen_sovi import model_trainer
        config_path = "./configs/mp_sovi.yml"
    elif args.trainer == "trans_so":
        from sophie.trainers.trainer_gen_trans_so import model_trainer
        config_path = "./configs/mp_trans_so.yml"
    elif args.trainer == "trans_sovi":
        from sophie.trainers.trainer_gen_trans_sovi import model_trainer
        config_path = "./configs/mp_trans_sovi.yml"
    elif args.trainer == "settrans":
        from sophie.trainers.trainer_gen_transset import model_trainer
        config_path = "./configs/mp_settrans.yml" # "settransgoal"
    elif args.trainer == "settransgoal":
        from sophie.trainers.trainer_gen_transset_goal import model_trainer
        config_path = "./configs/mp_settransgoal.yml"
    elif args.trainer == "soconf_goals":
        from sophie.trainers.trainer_gen_soconf_goals import model_trainer
        config_path = "./configs/mp_soconf_goals.yml"
    elif args.trainer == "soconf_goals_cgh":
        from sophie.trainers.trainer_gen_soconf_goals_cgh import model_trainer
        config_path = "./configs/mp_soconf_goals_cgh.yml"
    else:
        assert 1==0, "Error"

    trainer = model_trainer

    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    # with open(r'./configs/sophie_argoverse.yml') as config_file:
    with open(config_path) as config_file:
        config_file = yaml.safe_load(config_file)

        # Fill some additional dimensions

        config_file["base_dir"] = BASE_DIR
        exp_path = os.path.join(config_file["base_dir"], config_file["hyperparameters"]["output_dir"])   
        route_path = exp_path + "/config_file.yml"

        if not os.path.exists(exp_path):
            print("Create experiment path: ", exp_path)
            os.makedirs(exp_path) # makedirs creates intermediate folders

        with open(route_path,'w') as yaml_file:
            yaml.dump(config_file, yaml_file, default_flow_style=False)

        config_file = Prodict.from_dict(config_file)

    now = datetime.now()
    time = now.strftime("%H:%M:%S")

    logger = create_logger(os.path.join(exp_path, f"{config_file.dataset_name}_{time}.log"))
    logger.info("Config file: {}".format(config_path))
    
    trainer(config_file, logger)