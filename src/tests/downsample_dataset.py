#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Downsample images in given dataset. 
# Arguments   : Path to HR images directory.  
#               Scaling factor (int). 
#               Flag - Is last dimension color (True) or to be reduced (False)? 
# =============================================================================
import imageio
import glob
import numpy as np
import os
import sys
import skimage

# Get directory of images from input arguments. 
if not len(sys.argv) >= 4: 
    raise ValueError("Please state [directory, scale, is_multichannel] !")
directory = sys.argv[1]
scale = int(sys.argv[2])
is_multichannel = str(sys.argv[3]).lower() == "true"
if not directory[-1] == "/": 
    directory = directory + "/"
assert os.path.isdir(directory)
assert scale > 0 and scale % 2 == 0

# Iterate over all files in directory, load, downsample and save them. 
directory_out = directory.split("/")[-2].replace("HR", "LR_bicubic")
directory_out = os.path.dirname(directory[:-1]) + "/" + directory_out
os.makedirs(directory_out, exist_ok=True)
for filepath in glob.iglob(directory + "*.png"):
    hr = imageio.imread(filepath)
    lr = skimage.transform.rescale(hr, 1.0/scale, 
            anti_aliasing=True, multichannel=is_multichannel, mode='constant')
    filename, _ = os.path.splitext(os.path.basename(filepath))
    filename = directory_out + "/{}x{}.png".format(filename, scale)
    imageio.imwrite(filename, lr.astype(np.uint8))

