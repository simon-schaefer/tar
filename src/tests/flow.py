#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imageio
import numpy as np

from tar.miscellaneous import convert_flow_to_color

prev = imageio.imread("ressources/1_1.png")
prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
curr = imageio.imread("ressources/1_2.png")
curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.9, 15, 20, 100, 10, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

rgb = convert_flow_to_color(flow)
imageio.imsave("/Users/sele/Desktop/test.png", rgb)
