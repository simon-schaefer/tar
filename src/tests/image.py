#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Test cases for image domain. 
# =============================================================================
import unittest

from super_resolution.tools import image_toolbox as img_tools
from visual_test import *

IMAGE = "../../data/DIV2K_valid_HR/0801.png"

class TestImageToolbox(unittest.TestCase):

    def test_downsampling(self): 
        image = img_tools.load_image(IMAGE)
        image_hr, image_lr = img_tools.downsample(image, factor=2)
        img_tools.save(image_lr)

if __name__ == '__main__':
    unittest.main()