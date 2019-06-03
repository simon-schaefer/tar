#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Load and compare loss curves from two out directories.
# Arguments   : Out directiory path I.
#               Out directiory path II.
# =============================================================================
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils import parse_logging, save_path

# Parse out directories from user input.
parser = argparse.ArgumentParser(description="loss_curves")
parser.add_argument("--out1", type=str, default="")
parser.add_argument("--out2", type=str, default="")
parser.add_argument("--tag1", type=str, default="1")
parser.add_argument("--tag2", type=str, default="2")
args = parser.parse_args()
out1 = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], args.out1)
out2 = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], args.out2)
assert os.path.isdir(out1) and os.path.isdir(out2)

# Read and parse logging files to get loss curves.
log1 = parse_logging(os.path.join(out1, "log.txt"))
log2 = parse_logging(os.path.join(out2, "log.txt"))

# Plot loss curves and label with tags.
print("... plotting and saving loss curve")
fig = plt.figure()
plt.plot(log1["epoch"], log1["loss"], label=args.tag1)
plt.plot(log2["epoch"], log2["loss"], label=args.tag2)
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.savefig(save_path("loss_curves.png"))
plt.close()
