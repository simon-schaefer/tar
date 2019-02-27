#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Test cases for image domain. 
# =============================================================================
import numpy as np
import os
import unittest

import super_resolution.args as argus

class DataLoaderTest(unittest.TestCase): 
    
    def test_initialization(self): 
        args = argus.args
        loader = data.Data(args)
        assert loader
        loader_train = loader.loader_train
        loader_test = loader.loader_test
        assert loader_test, loader_train
    
    def test_batching(self): 
        args = argus.args
        loader = data.Data(args)
        loader_train = loader.loader_train
        for batch, (lr, hr, _) in enumerate(loader_train)
            print(batch)
            print(lr, hr)

if __name__ == '__main__':
    unittest.main()