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
import skimage

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int:
    """ Draws progress bar showing the number of executed
    iterations over the overall number of iterations.
    Increments the iteration and returns it. """
    status = ""
    progress = float(iteration) / float(num_steps)
    if progress >= 1.0:
        progress, status = 1.0, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (bar_length - block), round(progress*100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return iteration + 1

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
print("... found scales {}".format(scales))

# Iterate over all files in directory and check scale coherence.
image_files = glob.glob(hr_directory + "*.png")
num_files   = len(image_files)
error_files = []
for i, filepath in enumerate(image_files):
    hr = imageio.imread(filepath)
    filename, _ = os.path.splitext(os.path.basename(filepath))
    for s in scales:
        lr = imageio.imread(lr_directory + "X{}/{}x{}.png".format(s,filename,s))
        if not (hr.shape[0] == s*lr.shape[0]) and (hr.shape[1] == s*lr.shape[1]):
            error_files.append("{}_{}".format(filename, s))
    progress_bar(i, num_files)
print("... finished checking, error files: {}".format(error_files))
