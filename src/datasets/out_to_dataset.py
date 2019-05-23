#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Copy output images in directory of specific type to dataset
#               directory and rename them by removing the type term.
# Arguments   : Path to out directory.
#               HR-tag.
#               LR-tag.
# =============================================================================
import argparse
import glob
import os
import shutil
import sys

# Get directory of images from input arguments.
if not len(sys.argv) >= 3:
    raise ValueError("Please state [out_dir, HR_tag, LR_tag] !")
directory = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], sys.argv[1])
hr_tag, lr_tag = sys.argv[2], sys.argv[3]
assert os.path.isdir(directory)

# Create dataset standard form and get occurring fnames and scales.
out_dir = os.environ["SR_PROJECT_DATA_PATH"]
out_dir = os.path.join(out_dir,directory.split("/")[-1].replace("results_",""))
out_dir = out_dir + "_SR"
out_hr, out_lr = os.path.join(out_dir,"HR"), os.path.join(out_dir,"LR_bicubic")
os.makedirs(out_hr, exist_ok=True)
os.makedirs(out_lr, exist_ok=True)
fnames = []
for fname in glob.glob(os.path.join(directory, "*.png")):
    if fname.count(hr_tag)>0: fnames.append(os.path.basename(fname).split("_")[0])
scales = []
for fname in glob.glob(os.path.join(directory, "*.png")):
    sc_tag = int(os.path.basename(fname).split("_")[-2].replace("x",""))
    if not sc_tag in scales: scales.append(sc_tag)
for sc in scales: os.makedirs(os.path.join(out_lr, "X"+str(sc)),exist_ok=True)

# Copy files in fnames to out directory.
for fname in fnames:
    for sc in scales:
        hr_src = os.path.join(directory, fname+"_x"+str(sc)+"_"+hr_tag+".png")
        hr_dst = os.path.join(out_hr, fname+".png")
        shutil.copy2(hr_src, hr_dst)
        lr_src = os.path.join(directory, fname+"_x"+str(sc)+"_"+lr_tag+".png")
        lr_dst = os.path.join(out_lr, "X"+str(sc), fname+"x"+str(sc)+".png")
        shutil.copy2(lr_src, lr_dst)
