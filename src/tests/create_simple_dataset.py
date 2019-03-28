#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Create test dataset consisting of similar and repeated shapes. 
# =============================================================================
import imageio
import numpy as np
import os
import sys

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int: 
    """ Draws progress bar showing the number of executed 
    iterations over the overall number of iterations. 
    Increments the iteration and returns it. """
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

directory_out = os.path.join(os.environ["SR_PROJECT_DATA_PATH"], "SIMPLE")
os.makedirs(directory_out, exist_ok=True)
directory_out = os.path.join(directory_out, "HR")
os.makedirs(directory_out, exist_ok=True)
print("Starting dataset creation to {} ...".format(directory_out))
N = 100
for i in range(N): 
    img = np.ones((1024,960,3), dtype=np.uint8)*255
    x_start = np.random.randint(100, 800)
    y_start = np.random.randint(100, 700)
    width   = np.random.randint(50, 200)
    height  = np.random.randint(50, 200)
    channel = np.random.choice([0,1,2])
    img[x_start:x_start+width,y_start:y_start+height,channel] = 0
    filename = os.path.join(directory_out, str(i)+".png")
    imageio.imwrite(filename, img)
    progress_bar(i, N)
print("... finished dataset creation !")
