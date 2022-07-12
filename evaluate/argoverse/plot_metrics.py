#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Plot the metrics distribution (boxplot and histogram)
## in order to analyze the error

"""
Created on Fri Feb 25 12:19:38 2022
@author: Carlos Gómez-Huélamo, Miguel Eduardo Ortiz Huamaní and Marcos V. Conde
"""

# General purpose imports

import argparse
import os
import sys
import yaml
import pdb
import time
import glob
import csv

from pathlib import Path
from prodict import Prodict

# DL & Math imports

import pandas as pd
import numpy as np

# Plot imports

import matplotlib.pyplot as plt

#######################################

filepath = "results/social_lstm_mhsa/100.0_percent/exp1/val/metrics.csv"

df = pd.read_csv(filepath,sep=" ")
ade_column = df["ADE"][:-2]
fde_column = df["FDE"][:-2]

fig, ax = plt.subplots(figsize=(6,6), facecolor="white")

plt.xlabel("[m]")
plt.ylabel("Num sequences")
plt.xticks(np.arange(0, ade_column.max(), 2))
plt.title("ADE (Average Displacement Error, [m]) distribution")
plt.hist(ade_column,bins=100)
plt.savefig("results/social_lstm_mhsa/100.0_percent/exp1/val/ade.png", 
             bbox_inches='tight', 
             facecolor=fig.get_facecolor(), 
             edgecolor='none', 
             pad_inches=0)

plt.close("all")

plt.xlabel("[m]")
plt.ylabel("Num sequences")
plt.xticks(np.arange(0, fde_column.max(), 2))
plt.title("FDE (Final Displacement Error, [m]) distribution")
plt.hist(fde_column,bins=100)
plt.savefig("results/social_lstm_mhsa/100.0_percent/exp1/val/fde.png", 
             bbox_inches='tight', 
             facecolor=fig.get_facecolor(), 
             edgecolor='none', 
             pad_inches=0)