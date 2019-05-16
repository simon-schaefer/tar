#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Check downsamples dataset by comparing the LR shape with the
#               HR shape given the scale factor.
# Arguments   : Path to HR images directory.
#               Path to LR images directories (format X2).
# =============================================================================
import imageio
import glob
import numpy as np
import os
import sys

from utils import progress_bar

# Get directory of images from input arguments.
if not len(sys.argv) >= 2:
    raise ValueError("Please state [HR_directory, LR_directory] !")
hr_directory = sys.argv[1]
lr_directory = sys.argv[2]
if not hr_directory[-1] == "/": hr_directory = hr_directory + "/"
assert os.path.isdir(hr_directory)
if not lr_directory[-1] == "/": lr_directory = lr_directory + "/"
assert os.path.isdir(lr_directory)

print("Starting checking operation ...")
# Get all scales (assuming having directory name "Xs").
scales = [int(s.split("/")[-2][1:]) for s in glob.glob(lr_directory+"*/")]
max_scale = max(scales)
print("... found scales {}".format(scales))

# Iterate over all files in directory and check scale coherence.
image_files = glob.glob(hr_directory + "*.png")
num_files   = len(image_files)
error_files = []
error = ""
for i, hr_filepath in enumerate(image_files):
    hr = imageio.imread(hr_filepath)
    filename, _ = os.path.splitext(os.path.basename(hr_filepath))
    if hr.shape[0] % max_scale != 0 or hr.shape[1] % max_scale != 0:
        os.remove(hr_filepath)
        if hr.shape[0] % max_scale != 0:
            hr = hr[:max_scale*(hr.shape[0]//max_scale),:,:]
        if hr.shape[1] % max_scale != 0:
            hr = hr[:,:max_scale*(hr.shape[1]//max_scale),:]
        imageio.imwrite(hr_filepath, hr.astype(np.uint8))
    for s in scales:
        lr_filepath = lr_directory + "X{}/{}x{}.png".format(s,filename,s)
        lr = imageio.imread(lr_filepath)
        if not ((hr.shape[0] == s*lr.shape[0]) and (hr.shape[1] == s*lr.shape[1])):
            if filename not in error_files:
                error += "{},{},{},{}\n".format(filename,hr.shape,lr.shape,s)
                error_files.append(filename)
    progress_bar(i+1, num_files)
print("... finished checking, with error messages: \n{}".format(error))

# Delete errorerous files in all scales.
for file in error_files:
    hr_filepath = hr_directory + file + ".png"
    if os.path.exists(hr_filepath): os.remove(hr_filepath)
    for s in scales:
        lr_filepath = lr_directory + "X{}/{}x{}.png".format(s,file,s)
        if os.path.exists(lr_filepath): os.remove(lr_filepath)
print("... deleted errerous files.")
