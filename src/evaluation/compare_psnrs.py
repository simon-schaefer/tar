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
parser.add_argument("--index", type=str, default="SHRm&SLRm",
                    help="SHRb, SHRm, SLRb, SLRm")
parser.add_argument("--throw", type=str, default="")
args = parser.parse_args()
outs, tags = args.outs.split("&"), args.tags.split("&")
index_dict = {"SHRb": 0, "SHRm": 1, "SLRb": 2, "SLRm": 3}
value_labels = args.index.split("&")
value_indexs = [index_dict[x] for x in args.index.split("&")]
assert len(outs) == len(tags)
outs = [os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], x) for x in outs]
assert all([os.path.isdir(x) for x in outs])

# Read and parse logging files to get loss curves.
configs = [parse_config(os.path.join(x, "config.txt")) for x in outs]
logs    = [torch.load(os.path.join(x, "psnr_log.pt")) for x in outs]
log_lists = [build_log_list(x["data_valid"],x["scales_valid"]) for x in configs]
unique_dsets = np.unique([x for y in log_lists for x in y])
print("... unique datsets are {}".format(unique_dsets))
unique_all_dsets = []
for label in unique_dsets:
    if not all([label in x for x in log_lists]): continue
    unique_all_dsets.append(label)
print("... all contain datasets {}".format(unique_all_dsets))
unique_all_dsets = [x for x in unique_all_dsets if not args.throw in x]
print("... after filtering datasets {}".format(unique_all_dsets))

# Plot loss curves and label with tags.
print("... plotting and saving psnr curves")
num_uni_all = len(unique_all_dsets)
fig, ax = plt.subplots(1, num_uni_all, figsize=(5*num_uni_all,5))
for il, dset in enumerate(unique_all_dsets):
    for i in range(len(logs)):
        xs = np.arange(logs[i].size()[0])
        for j in range(len(value_indexs)):
            index = value_indexs[j]
            legend = "{}({})".format(tags[i], value_labels[j])
            ys = logs[i][:,log_lists[i].index(dset),index].tolist()
            ax[il].plot(xs, ys, label=legend)
    ax[il].set_title(dset)
    ax[il].legend()
    ax[il].set_xlabel("EPOCH")
    ax[il].set_ylabel("PSNR [dB]")
plt.savefig(save_path("psnr_curves.png"))
plt.close()
