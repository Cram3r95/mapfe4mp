#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Plot the metrics distribution (boxplot and histogram)
## in order to analyze the error

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import argparse
import pdb
import git
import sys
import os

# DL & Math imports

import pandas as pd
import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

#######################################

def plot_metrics_distribution(file_csv):
    """
    """

    root_dir = os.path.join(*file_csv.split('/')[:-1])

    df = pd.read_csv(file_csv,sep=" ")
    
    ade_column = df["ADE"][:-2].astype(float)
    fde_column = df["FDE"][:-2].astype(float)

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")

    # Plot ADE (Average Displacement Error) distribution

    plt.xlabel("[m]")
    plt.ylabel("Num sequences")

    plt.xticks(np.arange(0, ade_column.max(), 2))
    plt.title("ADE (Average Displacement Error, [m]) distribution")
    plt.hist(ade_column,bins=100)

    plt.savefig(os.path.join(root_dir,"ade_distribution.png"), 
                bbox_inches='tight', 
                facecolor=fig.get_facecolor(), 
                edgecolor='none', 
                pad_inches=0)

    plt.close("all")

    # Plot FDE (Final Displacement Error) distribution

    plt.xlabel("[m]")
    plt.ylabel("Num sequences")
    plt.xticks(np.arange(0, fde_column.max(), 2))
    plt.title("FDE (Final Displacement Error, [m]) distribution")
    plt.hist(fde_column,bins=100)
    plt.savefig(os.path.join(root_dir,"fde_distribution.png"), 
                bbox_inches='tight', 
                facecolor=fig.get_facecolor(), 
                edgecolor='none', 
                pad_inches=0)

if __name__ == "__main__":
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_csv", required=True, type=str)
    args = parser.parse_args()
    print("args: ", args)

    plot_metrics_distribution(args.file_csv)