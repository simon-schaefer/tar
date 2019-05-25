#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Network architecture visualization.
# =============================================================================
import torch
from torch.autograd import Variable
import tar.models as tar_models
import tar.models.aetad

model = tar_models.aetad.AETAD()
dummy_input = Variable(torch.randn(4, 3, 64, 64))
torch.onnx.export(model, dummy_input, "aetad.onnx")
