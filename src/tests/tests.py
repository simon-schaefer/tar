#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Test cases for image domain. 
# =============================================================================
import imageio
import numpy as np
import os
import unittest

import torch
from torch import nn

import tar.inputs as argus
import tar.dataloader as dataloader
import tar.optimization as optimization
import tar.miscellaneous as miscellaneous
import tar.modules as modules

class DataLoaderTest(unittest.TestCase): 
    
    def test_initialization(self): 
        args = argus.args
        loader = dataloader._Data_(args)
        assert loader
        # Testing data loader. 
        loader_test = loader.loader_test
        assert loader_test
        # Training data loader. 
        if not args.valid_only: 
            assert loader.loader_train
    
    def test_batching(self): 
        args = argus.args
        args.valid_only = False
        args.patch_size, args.scales_train = 96, [2,4]
        loader = dataloader._Data_(args)
        for scale, d in loader.loader_train.items(): 
            for batch, (lr, hr, files) in enumerate(d): 
                assert lr.shape[0] == args.batch_size and hr.shape[0] == args.batch_size
                assert lr.shape[1] == args.n_colors and hr.shape[1] == args.n_colors
                s, ls = args.patch_size, int(args.patch_size/scale)
                assert lr.shape[2] == ls and hr.shape[2] == s
                assert lr.shape[3] == ls and hr.shape[3] == s
                break

    def test_div2k(self):
        args = argus.args
        args.valid_only = False
        args.data_train = "DIV2K"
        args.data_test = "DIV2K"
        args.patch_size, args.scale = 96, 2
        loader = dataloader._Data_(args)
        for batch, (lr, hr, files) in enumerate(loader.loader_train[2]): 
            assert lr.shape[0] == args.batch_size and hr.shape[0] == args.batch_size
            assert lr.shape[1] == args.n_colors and hr.shape[1] == args.n_colors
            s, ls = args.patch_size, int(args.patch_size/args.scale)
            assert lr.shape[2] == ls and hr.shape[2] == s
            assert lr.shape[3] == ls and hr.shape[3] == s
            break       

class MiscellaneousTest(unittest.TestCase): 

    def test_timer(self): 
        timer = miscellaneous._Timer_()
        timer.hold()
        dt = timer.toc()
        assert dt > 0 and dt < 1.0

    def test_checkpoint(self): 
        args = argus.args
        ckp = miscellaneous._Checkpoint_(args)
        ckp.write_log("test")
        ckp.done()

class OptimizationTest(unittest.TestCase):

    def test_loss_init(self): 
        # Intialize loss module input arguments. 
        args = argus.args
        args.load = ""
        ckp = miscellaneous._Checkpoint_(args)
        # Initialize loss module. 
        loss = optimization._Loss_(args, ckp)  
        assert loss
        ckp.done()

    def test_loss_forward(self): 
        # Intialize loss module input arguments. 
        args = argus.args
        args.valid_only = False
        args.loss = "HR*1*L1"
        args.load = ""
        args.data_train = "DIV2K"
        args.data_test = "DIV2K"
        args.scales_train = 2
        ckp = miscellaneous._Checkpoint_(args)
        loader = dataloader._Data_(args)
        loader_train = loader.loader_train   
        # Test forward. 
        loss = optimization._Loss_(args, ckp)  
        loss.start_log()   
        for batch, (lr, hr, files) in enumerate(loader_train[2]): 
            loss_kwargs = {'HR_GT': hr, 'HR_OUT': hr}
            loss_sum = loss.forward(loss_kwargs)
            assert loss_sum == 0
            break 
        ckp.done()

    def test_loss_display(self): 
        # Intialize loss module input arguments. 
        args = argus.args
        args.loss = "HR*1*L1"
        args.load = ""
        ckp = miscellaneous._Checkpoint_(args)
        # Test loss description. 
        loss = optimization._Loss_(args, ckp)  
        loss.start_log()      
        log = loss.display_loss(0)
        log = str(log)
        assert log.find("TOTAL") > 0 and log.find("HR") > 0
        ckp.done()

    def test_loss_modules(self): 
        # Intialize loss module input arguments. 
        args = argus.args
        args.loss = "HR*1*L1"
        args.load = ""
        ckp = miscellaneous._Checkpoint_(args)
        # Test loss description. 
        loss = optimization._Loss_(args, ckp)        
        modules = loss.get_loss_module()  
        assert modules     
        ckp.done()

class ModulesTest(unittest.TestCase): 

    class PixelShuffle_Forward_Backward(nn.Module): 

        def __init__(self, factor):
            super(ModulesTest.PixelShuffle_Forward_Backward, self).__init__()
            self._seq = nn.Sequential(
                modules._ReversePixelShuffle_(downscale_factor=factor), 
                nn.PixelShuffle(upscale_factor=factor)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor: 
            return self._seq(x)

    def test_inv_pixel_shuffle(self): 
        model = self.PixelShuffle_Forward_Backward(factor=2)
        x = torch.arange(16).view(1,1,4,4)
        y = model(x)
        assert torch.all(torch.eq(x, y))
        x = torch.arange(48).view(1,3,4,4)
        y = model(x)
        assert torch.all(torch.eq(x, y))

class PSNRTest(unittest.TestCase): 

    def test_psnr(self): 
        presults = os.environ["SR_PROJECT_PROJECT_HOME"] + "/src/tests/"
        shr = imageio.imread(presults+"/ressources/SHR.png")
        shr = torch.from_numpy(shr).double()/255.0
        hr  = imageio.imread(presults+"/ressources/HR.png")
        hr  = torch.from_numpy(hr).double()/255.0
        psnr = miscellaneous.calc_psnr(shr, hr, patch_size=None, rgb_range=1.0)
        assert np.abs(psnr - 38.7728) < 1e-4


if __name__ == '__main__':
    unittest.main()