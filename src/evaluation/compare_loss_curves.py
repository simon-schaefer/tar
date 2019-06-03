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

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils import parse_logging, save_path

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
logs = [parse_logging(os.path.join(x, "log.txt")) for x in outs]

# Plot loss curves and label with tags.
print("... plotting and saving loss curve")
fig = plt.figure()
for log, tag in zip(logs, tags):
    plt.plot(log["epoch"], log["loss"], label=tag)
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.savefig(save_path("loss_curves.png"))
plt.close()
