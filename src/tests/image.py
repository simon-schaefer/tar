#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Test cases for image domain. 
# =============================================================================
import numpy as np
import os
import unittest

from super_resolution.tools import image_toolbox as img_tools
from visual_test import *

IMAGE  = os.environ['SR_PROJECT_DATA_PATH'] + "/DIV2K_valid_HR/0801.png"
OUTPUT = os.environ['SR_PROJECT_OUTS_PATH']

class TestImageToolbox(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()