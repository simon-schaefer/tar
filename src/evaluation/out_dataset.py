#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Create and save dataframe by scrapping out directories.
# Arguments   : Filter (format="key1:value1&key2:value2...")
#               Directory in out path.
# =============================================================================
import argparse
import numpy as np
import os
import pandas as pd

from utils import scrap_outputs, save_path

parser = argparse.ArgumentParser(description="psnr_time")
parser.add_argument("--directory", type=str, default="")
parser.add_argument("--filter", type=str, default="")
parser.add_argument("--sort", type=str, default="")
args = parser.parse_args()

print("Scrapping and filtering outs data ...")
dir = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], args.directory)
data = scrap_outputs(directory=dir)
data["RUNTIME_AL"] = round(data["RUNTIME_AL"]*1000, 2)
data["RUNTIME_DW"] = round(data["RUNTIME_DW"]*1000, 2)
data["RUNTIME_UP"] = round(data["RUNTIME_UP"]*1000, 2)
for key_value in args.filter.split("&"):
    if key_value == "": continue
    key, value = key_value.split(":")
    if len(value.split("/")) > 0:
        data = data[data[key].isin(value.split("/"))]
    else:
        data = data[data[key] == value]
print("... dropping pure NaN columns")
data = data.dropna(axis=1, how='all')
if args.sort != "":
    print("... sorting by value")
    data = data.sort_values(args.sort)
print("... saving dataframe to data_{}.csv".format(args.directory))
data.to_csv(save_path("data_{}.csv".format(args.directory)))
