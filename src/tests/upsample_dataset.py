#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Upsample all images in given dataset. 
# Input arg   : dataset name, scale factor (int).  
# =============================================================================
import os
import sys

import super_resolution.tools.files_toolbox as file_tools
import super_resolution.tools.image_toolbox as img_tools

# Check dataset and load dataset files.
assert len(sys.argv) >= 3
dataset_name = sys.argv[1]
scale_factor = int(sys.argv[2])
dataset_path = os.environ['SR_PROJECT_DATA_PATH'] + "/" + dataset_name
dataset = file_tools.load_files(dataset_path, "png")
print("Upsampling dataset with %d samples ..." % len(dataset))

# Upscale images and store in new folder. 
dataset_new_path = dataset_path + "_x" + str(scale_factor)
if not os.path.exists(dataset_new_path): 
    os.makedirs(dataset_new_path)
for sample_path in dataset: 
    img = img_tools.load_image(sample_path)
    _, img = img_tools.upsample(img, factor=scale_factor)
    img_name = sample_path.split("/")[-1]
    img_tools.save_image(img, dataset_new_path + "/" + img_name)