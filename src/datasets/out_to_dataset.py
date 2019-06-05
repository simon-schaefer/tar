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
dset_name = os.path.basename(out_dir)
out_hr, out_lr = os.path.join(out_dir,"HR"), os.path.join(out_dir,"LR_bicubic")
os.makedirs(out_hr, exist_ok=True)
os.makedirs(out_lr, exist_ok=True)
scales = []
for fname in glob.glob(os.path.join(directory, "*.png")):
    if os.path.basename(fname).count("EPS") > 0: continue 
    sc_tag = int(os.path.basename(fname).split("_")[-2].replace("x",""))
    if not sc_tag in scales: scales.append(sc_tag)
for sc in scales: os.makedirs(os.path.join(out_lr, "X"+str(sc)),exist_ok=True)
print(scales)
fnames = []
for fname in glob.glob(os.path.join(directory, "*.png")):
    if fname.count("_"+hr_tag+".png")>0:
        name = os.path.basename(fname).replace("_"+hr_tag+".png","")
        for scale in scales: name = name.replace("_x" + str(scale), "")
        fnames.append(name)
print(fnames)

# Copy files in fnames to out directory.
for fname in fnames:
    for sc in scales:
        hr_src = os.path.join(directory, fname+"_x"+str(sc)+"_"+hr_tag+".png")
        hr_dst = os.path.join(out_hr, fname+".png")
        shutil.copy2(hr_src, hr_dst)
        lr_src = os.path.join(directory, fname+"_x"+str(sc)+"_"+lr_tag+".png")
        lr_dst = os.path.join(out_lr, "X"+str(sc), fname+"x"+str(sc)+".png")
        shutil.copy2(lr_src, lr_dst)
print("... successfully created {} dataset !".format(dset_name))
