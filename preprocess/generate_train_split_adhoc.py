#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Select the most difficult sequences in the training split.
## Apply data augmentation offline (e.g. rotation) to increase the
## number of difficult trajectories (accelerations, curves, etc.)

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import yaml
from prodict import Prodict
import os
import git

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)