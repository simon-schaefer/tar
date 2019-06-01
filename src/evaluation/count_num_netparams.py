#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Count the number of network parameters.
# Arguments   : Model name
# =============================================================================
import argparse
from .utils import num_model_params

# Parse input arguments.
parser = argparse.ArgumentParser(description="net_params")
parser.add_argument("--model", type=str, default="")
args = parser.parse_args()
model_name = str(args.model)
# Determine number of model parameters and log it.
num_params = num_model_params(model_name)
print("Number of parameters of {} is {}".format(model_name, num_params))
