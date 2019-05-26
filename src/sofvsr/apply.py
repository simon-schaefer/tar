#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Apply SOFVSR model to given batch of images.
#               SOFVSR converts the current as well as the prev and next image
#               to a the YCbCr space and concatenates their Y axes as a network
#               input. Afterwards it scales up Cb and Cr using bilinear
#               interpolation and concatenates the predicted HR Y axis with the
#               interpolated Cb and Cr values. Finally, the image is converted
#               to RGB space.
# =============================================================================
import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

from sofvsr.modules import SOFVSR as SOFVSR_NET
from sofvsr.data_utils import ycbcr2rgb, rgb2ycbcr

class SOFVSR(object):

    def __init__(self, model, scale, use_gpu=True):
        # Load SOFVSR model from model path.
        self._net = SOFVSR_NET(upscale_factor=scale)
        model_path = os.path.join(os.environ["SR_PROJECT_MODELS_PATH"], "sofvsr")
        model_path = os.path.join(model_path, model + '.pth')
        ckpt = torch.load(model_path)
        self._net.load_state_dict(ckpt)
        if use_gpu: self._net.cuda()
        # Set auxialiary variables.
        self._use_gpu = use_gpu
        self._scale   = scale

    def apply(self, LR0, LR1, LR2):
        # Input checks.
        assert LR0.size() == LR1.size() == LR2.size()
        # Input preprocessing - Create Y cube image.
        LR0_y, LR1_y, LR2_y = self._to_y_image(LR0, LR1, LR2)
        LR_y_cube = torch.cat((LR0_y, LR1_y, LR2_y), 1)
        # Input preprocessing - Create Cb & Cr interpolation images.
        LR1_bicubic = torch.nn.functional.interpolate(LR1,
                        scale_factor=self._scale, mode='bilinear')
        _, SR_cb, SR_cr = self._rgb2ycbcr(LR1_bicubic)
        SR_cb, SR_cr = self._expand_dim(SR_cb), self._expand_dim(SR_cr)
        # Apply model to input and return outputs.
        LR_y_cube = Variable(LR_y_cube)
        if self._use_gpu: LR_y_cube = LR_y_cube.cuda()
        SR_y = self._net(LR_y_cube)
        SR_y = self._expand_dim(SR_y)
        # Image postprocessing.
        SR_ycbcr = torch.cat((SR_y, SR_cb, SR_cr), 1)
        SR_rgb = self._ycbcr2rgb(SR_ycbcr)
        return SR_rgb

    def _to_y_image(self, *tensors):
        def rgb_to_y(x):
          x, sz = Variable(x.data.new(*x.size())), x.size()
          x = x[:, 0, :, :]*65.481+x[:, 1, :, :]*128.553+x[:, 2, :, :]*24.966+16
          return self._expand_dim(x/255.0)
        return [rgb_to_y(x) for x in tensors]

    def _rgb2ycbcr(self, x):
        y  = 0.257*x[:, 0, :, :]+0.504*x[:, 1, :, :]+0.098*x[:, 2, :, :]+16/255.0
        cb = -0.148*x[:, 0, :, :]-0.291*x[:, 1, :, :]+0.439*x[:, 2, :, :]+128/255.0
        cr = 0.439*x[:, 0, :, :]-0.368*x[:, 1, :, :]-0.071*x[:, 2, :, :]+128/255.0
        return y, cb, cr

    def _ycbcr2rgb(self, x):
        img_r = 1.164*(x[:, 0, :, :]-16/255.0)+1.596*(x[:, 2, :, :]-128/255.0)
        img_r = self._expand_dim(img_r)
        img_g = 1.164*(x[:, 0, :, :]-16/255.0)-0.392*(x[:, 1, :, :]-128/255.0)
        img_g = img_g-0.813*(x[:, 2, :, :]-128/255.0)
        img_g = self._expand_dim(img_g)
        img_b = 1.164*(x[:, 0, :, :]-16/255.0)+2.017*(x[:, 1, :, :]-128/255.0)
        img_b = self._expand_dim(img_b)
        return torch.cat((img_r,img_g,img_b), 1)

    def _expand_dim(self, x):
        if len(x.size()) == 2: x = x.unsqueeze_(0)
        return x.unsqueeze_(0).view(x.size()[1],1,x.size()[2],x.size()[3])
