#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Crop to images for comparison at same location and save
#               overall image.
# =============================================================================
import imageio
import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Read input arguments and load both images.
if not len(sys.argv) >= 6:
    raise ValueError("Please state [img1, img2, x, y, w, h] !")
fimg1, fimg2 = sys.argv[1], sys.argv[2]
assert os.path.isfile(fimg1) and os.path.isfile(fimg2)
x, y = int(sys.argv[3]), int(sys.argv[4])
w, h = int(sys.argv[5]), int(sys.argv[6])
img1, img2 = imageio.imread(fimg1), imageio.imread(fimg2)
assert img1.shape == img2.shape
assert img1.shape[0] >= x + w and img1.shape[1] >= y + h
grayscale = len(img1.shape) == 2

# Crop images and draw overall plot.
if grayscale:
    img1c, img2c = img1[x:x+w, y:y+h], img2[x:x+w, y:y+h]
else:
    img1c, img2c = img1[x:x+w, y:y+h, :], img2[x:x+w, y:y+h, :]
fig = plt.figure()
if grayscale: plt.gray()
ax = fig.add_subplot(221)
plt.imshow(img1c)
plt.axis('off')
ax = fig.add_subplot(222)
plt.imshow(img2c)
plt.axis('off')
ax = fig.add_subplot(223)
plt.imshow(img1)
rect = patches.Rectangle((y,x),h,w,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.axis('off')
ax = fig.add_subplot(224)
rect = patches.Rectangle((y,x),h,w,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.imshow(img2)
plt.axis('off')

# Save overall figure in same location as first image.
fresult = os.path.join(os.path.dirname(fimg1), "result.png")
plt.savefig(fresult)
