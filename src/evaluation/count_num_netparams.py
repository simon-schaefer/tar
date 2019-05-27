#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Count the number of network parameters.
# =============================================================================
import tar
import tar.inputs
import torch

# Load model from normal framework.
tar.miscellaneous.print_header()
args   = tar.inputs.args
ckp    = tar.miscellaneous._Checkpoint_(args)
model  = tar.modules._Model_(args, ckp)

# Get torch module and return number of parameters.
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters is {}".format(num_params))
