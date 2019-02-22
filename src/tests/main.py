#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Test cases for image domain. 
# =============================================================================
import numpy as np
import os
import unittest

from super_resolution.image_domain import dataloader
from super_resolution.tools import files_toolbox as file_tools
from super_resolution.tools import image_toolbox as img_tools
from visual_test import *

IMAGE   = os.environ['SR_PROJECT_DATA_PATH'] + "/DIV2K_valid_HR/0801.png"
OUTPUT  = os.environ['SR_PROJECT_OUTS_PATH']

TRAIN_DATASET = os.environ['SR_PROJECT_DATA_PATH'] + "/DIV2K_train_HR"
VALID_DATASET = os.environ['SR_PROJECT_DATA_PATH'] + "/DIV2K_valid_HR"

class TestToolsImageToolbox(unittest.TestCase):

    def test_downsampling(self): 
        image = img_tools.load_image(IMAGE)
        img_tools.save_image(image, OUTPUT + "/original.png")
        image_hr, image_lr = img_tools.downsample(image, factor=2)
        img_tools.save_image(image_hr, OUTPUT + "/out_hr.png")
        img_tools.save_image(image_lr, OUTPUT + "/out_lr.png")

    def test_normalization(self): 
        image = img_tools.load_image(IMAGE)
        image = img_tools.normalize(image)
        assert not np.any(image > 0.5) or not np.any(image < -0.5)

    def test_random_sub_sample(self): 
        image = img_tools.load_image(IMAGE)
        subsample = img_tools.random_sub_sample(image, 96, 40)
        assert subsample.shape == (3, 96, 40)

class TestToolsFilesToolbox(unittest.TestCase): 

    def test_load_files(self): 
        files = file_tools.load_files(VALID_DATASET, "png")
        assert len(files) == 100

class TestImageDomainDataLoader(unittest.TestCase): 

    def test_next_batch(self): 
        loader = dataloader.DataLoader([TRAIN_DATASET], [VALID_DATASET], 
                    scale_guidance=2, batch_size=6, subsize=96)
        batch_hr, batch_lr, _ = loader.next_batch()
        assert batch_hr.shape == (6, 3, 96, 96)
        assert batch_lr.shape == (6, 3, 48, 48)

if __name__ == '__main__':
    unittest.main()