#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Downsample images in given dataset for RGB images. For
#               grayscale input the channels are stacked, i.e. R=G=B.
# Arguments   : Path to HR images directory.
#               Scaling factor (int).
#               Maximal scale factor (int).
#               Path to LR images directory (optional).
# =============================================================================
import imageio
import glob
import numpy as np
import os
import sys
import skimage

from utils import progress_bar

# Get directory of images from input arguments.
if not len(sys.argv) >= 3:
    raise ValueError("Please state [directory, scale] !")
directory = sys.argv[1]
scale = int(sys.argv[2])
max_scale = int(sys.argv[3])
if not directory[-1] == "/":
    directory = directory + "/"
directory_out = sys.argv[4] if len(sys.argv) > 4 else "LR_bicubic"
assert os.path.isdir(directory)
assert scale > 0 and scale % 2 == 0
assert max_scale > 0 and max_scale % 2 == 0

# Iterate over all files in directory, load, downsample and save them.
directory_out = os.path.dirname(directory[:-1]) + "/" + directory_out
os.makedirs(directory_out, exist_ok=True)
directory_out = directory_out + "/X{}".format(scale)
os.makedirs(directory_out, exist_ok=True)
print("Starting downscaling to {} ...".format(directory_out))
image_files = glob.glob(directory + "*.png")
num_files   = len(image_files)
for i, filepath in enumerate(image_files):
    hr = imageio.imread(filepath)
    if len(hr.shape) == 2:
        hr = np.stack((hr,hr,hr), axis=2)
    if hr.shape[0] % max_scale != 0:
        hr = hr[:max_scale*(hr.shape[0]//max_scale),:,:]
    if hr.shape[1] % max_scale != 0:
        hr = hr[:,:max_scale*(hr.shape[1]//max_scale),:]
    ldim = None
    lr = skimage.transform.rescale(hr, 1.0/scale,
            anti_aliasing=True,
            multichannel=True,
            mode='reflect',
            preserve_range=True,
            order=4
    )
    filename, _ = os.path.splitext(os.path.basename(filepath))
    filename = directory_out + "/{}x{}.png".format(filename,scale)
    imageio.imwrite(filename, lr.astype(np.uint8))
    progress_bar(i+1, num_files)
print("... finished downscaling !")
