#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Load and compare loss curves from two out directories.
# Arguments   : Out directiory paths (seperated by &).
#               Plot labels (tags) (seperated by &).
# =============================================================================
import argparse
import os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tar.miscellaneous import build_log_list
from utils import parse_config, save_path

# Parse out directories from user input.
parser = argparse.ArgumentParser(description="psnr_curves")
parser.add_argument("--outs", type=str, default="")
parser.add_argument("--tags", type=str, default="")
args = parser.parse_args()
outs, tags = args.outs.split("&"), args.tags.split("&")
assert len(outs) == len(tags)
outs = [os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], x) for x in outs]
assert all([os.path.isdir(x) for x in outs])

# Read and parse logging files to get loss curves.
configs = [parse_config(os.path.join(x, "config.txt")) for x in outs]
logs    = [torch.load(os.path.join(x, "psnr_log.pt")) for x in outs]
log_lists = [build_log_list(x["data_valid"],x["scales_valid"]) for x in configs]
unique_labels = np.unique([x for y in log_lists for x in y])
print("... unique labels are {}".format(unique_labels))
unique_all_labels = []
for label in unique_labels:
    if not all([label in x for x in log_lists]): continue
    unique_all_labels.append(label)
print("... all contain labels {}".format(unique_all_labels))

# Plot loss curves and label with tags.
print("... plotting and saving psnr curves")
fig, ax = plt.subplots(1, len(unique_all_labels), figsize=(15,5))
for il, label in enumerate(unique_all_labels):
    for i in range(len(logs)):
        xs = np.arange(logs[i].size()[0])
        ys = logs[i][:,log_lists[i].index(label),0].tolist()
        ax[il].plot(xs, ys, label=tags[i])
    ax[il].legend()
    ax[il].set_xlabel("EPOCH")
    ax[il].set_ylabel("PSNR [dB]")
    ax[il].set_title(label)
plt.savefig(save_path("psnr_curves.png"))
plt.close()
