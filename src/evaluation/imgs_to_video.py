#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Build video from images using OpenCV.
# =============================================================================
import cv2
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Read input arguments.
if not len(sys.argv) >= 2:
    raise ValueError("Please state [imgs_path, video_name, img_tag] !")
path_img_dir = sys.argv[1]
assert os.path.isdir(path_img_dir)
video_name = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else None

# Make list of images in img_dir having the tag.
images = glob.glob(path_img_dir + "/*" + ".png")
for x in range(0, len(images)): images[x] = path_img_dir+"/hr"+str(x)+"x4.png"
if tag is not None: images = [x for x in images if x.count(tag) > 0]
print(images)

# Create video using OpenCV functionalities.
fps = 10
frame = cv2.imread(images[0])
height, width, _ = frame.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))
for image in images: video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()
