#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Build video from images using OpenCV.
# Arguments   : Path to images (directory)
#               Output video name
#               Image tag (optional, if multiple kinds of images in directory)
# =============================================================================
import argparse
import cv2
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Read input arguments.
parser = argparse.ArgumentParser(description="imgs_to_video")
parser.add_argument("--directory", type=str, default="")
parser.add_argument("--video_name", type=str, default="video.avi")
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()
assert os.path.isdir(args.directory)
tag = "_" + args.tag if args.tag != "" else None

# Make list of images in img_dir having the tag.
images = glob.glob(args.directory + "/*" + ".png")
for x in range(0, len(images)): images[x] = args.directory+"/hr"+str(x)+"x4_x4b.png"
#images = sorted(images)
if tag is not None: images = [x for x in images if x.count(tag) > 0]

# Create video using OpenCV functionalities.
fps = 10
frame = cv2.imread(images[0])
height, width, _ = frame.shape
video = cv2.VideoWriter(args.video_name, 0, fps, (width,height))
for image in images:
    print(image)
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()
