#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Compare two images of images of same scale by determining the
#               PSNR value on a local level. Plot result as heatmap.
# Arguments   : Path to HR image.
#               Path to LR image.
# =============================================================================
import imageio
import numpy as np
import os
from skimage.color import rgb2gray
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from tar.miscellaneous import calc_psnr

if not len(sys.argv) >= 2: raise ValueError("Please state [IMG1, IMG2] !")
img1_file, img2_file = sys.argv[1], sys.argv[2]
assert os.path.isfile(img1_file) and os.path.isfile(img2_file)

# Parameters.
patch_size = 8

# Load images and check identity of dimensions.
img1, img2 = imageio.imread(img1_file), imageio.imread(img2_file)
assert img1.shape == img2.shape
w,h,_ = img1.shape
img1 = img1[:w//patch_size*patch_size,:h//patch_size*patch_size,:]
img2 = img2[:w//patch_size*patch_size,:h//patch_size*patch_size,:]
w,h,_ = img1.shape
assert w % patch_size == 0 and h % patch_size == 0
# Iterate over image and determine the PSNR value patchwise, store all
# psnr values in numpy array (in patch location).
psnrs_img = np.zeros((w,h))
for iw in range(w//patch_size):
    for ih in range(h//patch_size):
        ws,wt = iw*patch_size, (iw+1)*patch_size
        hs,ht = ih*patch_size, (ih+1)*patch_size
        p1t = torch.from_numpy(img1[ws:wt,hs:ht,:])
        p2t = torch.from_numpy(img2[ws:wt,hs:ht,:])
        psnrs_img[ws:wt,hs:ht] = calc_psnr(p1t, p2t)

# Postprocess psnr values array.
pnsr_std, psnr_mean = np.std(psnrs_img), np.mean(psnrs_img)
psnrs_img[psnrs_img > psnr_mean + 2*pnsr_std] = psnr_mean + 2*pnsr_std

# Plot results.
plt.subplot(1,2,1)
plt.imshow(img1, cmap='viridis')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(psnrs_img, alpha=0.7, cmap='viridis')
plt.colorbar()
plt.axis('off')
img1_file, _ = os.path.splitext(os.path.basename(img1_file))
img2_file, _ = os.path.splitext(os.path.basename(img2_file))
filename = "fusion_{}_{}.png".format(img1_file, img2_file)
plt.savefig(os.path.join(os.environ['SR_PROJECT_OUTS_PATH'],filename))
plt.close()
