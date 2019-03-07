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

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int: 
    ''' Draws progress bar showing the number of executed 
    iterations over the overall number of iterations. 
    Increments the iteration and returns it. '''
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
print("Starting downscaling to {} ...".format(directory_out))
image_files = glob.glob(directory + "*.png")
num_files   = len(image_files)
for i, filepath in enumerate(image_files):
    hr = imageio.imread(filepath)
    ldim = None
    lr = skimage.transform.rescale(hr, 1.0/scale,  
            anti_aliasing=True, 
            multichannel=is_multichannel, 
            mode='reflect', 
            preserve_range=True, 
            order=4
    )
    filename, _ = os.path.splitext(os.path.basename(filepath))
    filename = directory_out + "/{}x{}.png".format(filename, scale)
    imageio.imwrite(filename, lr.astype(np.uint8))
    progress_bar(i, num_files)
print("... finished downscaling !")