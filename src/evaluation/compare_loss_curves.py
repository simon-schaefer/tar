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

from utils import parse_logging

# Parse out directories from user input.
parser = argparse.ArgumentParser(description="loss_curves")
parser.add_argument("--out1", type=str, default="")
parser.add_argument("--out2", type=str, default="")
parser.add_argument("--tag1", type=str, default="1")
parser.add_argument("--tag2", type=str, default="2")
args = parser.parse_args()
assert os.path.isdir(args.out1) and os.path.isdir(args.out2)

# Read and parse logging files to get loss curves.
log1 = parse_logging(os.path.join(args.out1, "log.txt"))
log2 = parse_logging(os.path.join(args.out2, "log.txt"))

# Plot loss curves and label with tags.
fig = plt.figure()
plt.plot(log1["epoch"], log1["loss"], legend=args.tag1)
plt.plot(log2["epoch"], log2["loss"], legend=args.tag2)
plt.legends()
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.savefig("loss_curves.png")
plt.close()
