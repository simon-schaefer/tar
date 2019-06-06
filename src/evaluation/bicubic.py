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
import glob
import numpy as np
import os
import skimage

# Get directory of images from input arguments.
parser = argparse.ArgumentParser(description="bicubic")
parser.add_argument("--directory", type=str, default="")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--scale", type=int, default=4)
args = parser.parse_args()
assert os.path.isdir(args.directory)
assert args.scale > 0 and args.scale % 2 == 0

# Load images from directory.
lrs = glob.glob(args.directory + "/*.png")
if args.tag != "": lrs = [x for x in lrs if x.count(args.tag) > 0]
out = "{}_x{}b".format(os.path.basename(args.directory), args.scale)
out = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], out)
os.makedirs(out)

# Load image, upscale it and save it.
print("Bilinearly upscaling {} ...".format(os.path.basename(args.directory)))
for fname_lr in lrs:
    lr = imageio.imread(fname_lr)
    lr = lr[:,:,:3]
    hr = skimage.transform.rescale(lr, args.scale,
            anti_aliasing=True,
            multichannel=True,
            mode='reflect',
            preserve_range=True,
            order=3
    )
    filename, _ = os.path.splitext(os.path.basename(fname_lr))
    filename    = filename.replace(".png","")+"_x{}b.png".format(str(args.scale))
    imageio.imwrite(os.path.join(out,filename), hr.astype(np.uint8))
print("... finished bilinear upscaling !")
