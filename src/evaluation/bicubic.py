#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Upsample image bilinearly using skimage rescale transformation.
# Arguments   : Path to LR image
#               Scaling factor
# =============================================================================
import argparse
import imageio
import numpy as np
import os
import skimage

# Get directory of images from input arguments.
parser = argparse.ArgumentParser(description="bicubic")
parser.add_argument("--lr", type=str, default="")
parser.add_argument("--scale", type=int, default=4)
args = parser.parse_args()
assert os.path.isfile(args.lr)
assert args.scale > 0 and args.scale % 2 == 0

# Load image, upscale it and save it.
print("Bilinearly upscaling {} ...".format(path_LR))
lr = imageio.imread(args.lr)
lr = lr[:,:,:3]
hr = skimage.transform.rescale(lr, args.scale,
        anti_aliasing=True,
        multichannel=True,
        mode='reflect',
        preserve_range=True,
        order=3
)
filename, _ = os.path.splitext(os.path.basename(args.lr))
filename    = filename.replace(".png","")+"_x{}b.png".format(str(args.scale))
dirname = os.path.dirname(args.lr)
path_HR = os.path.join(dirname,filename)
imageio.imwrite(path_HR, hr.astype(np.uint8))
print("... finished bilinear upscaling !")
