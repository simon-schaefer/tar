#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Determine PSNR value between two directories.
# =============================================================================
import argparse
import imageio
import glob
import numpy as np
import os
import torch

from tar.miscellaneous import calc_psnr

# Get directory of images from input arguments.
parser = argparse.ArgumentParser(description="psnr_average")
parser.add_argument("--base", type=str, default="")
parser.add_argument("--target", type=str, default="")
parser.add_argument("--target_drop", type=str, default="")
args = parser.parse_args()
assert os.path.isdir(args.base) and os.path.isdir(args.target)

# Load images from directory.
imgs_base   = glob.glob(args.base + "/*.png")
imgs_target = glob.glob(args.target + "/*.png")

# Load image, upscale it and save it.
print("Determining PSNR values ...")
psnrs = []
for fname in imgs_base:
    base = imageio.imread(fname)
    if len(base.shape) >= 3: base = base[:,:,:3]
    fname_base = os.path.join(args.target, os.path.basename(fname))
    fname_base = fname_base.replace(".png", args.target_drop + ".png")
    target = imageio.imread(fname_base)
    if target.shape[2] == 2: target = np.vstack((target,target,target))
    if len(target.shape) >= 3 and len(base.shape) == 2: target = target[:,:,0]
    base_tensor   = torch.from_numpy(base/255.0)
    target_tensor = torch.from_numpy(target/255.0)
    psnr = calc_psnr(base_tensor, target_tensor, rgb_range=1.0)
    psnrs.append(psnr)
print("... mean psnr value = {}".format(np.mean(psnrs)))
