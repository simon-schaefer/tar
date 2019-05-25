#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Utility functions for evaluations.
# =============================================================================
import os
import pandas as pd

def read_config_file(fname):
    d = {}
    with open(fname) as f:
        for line in f:
            split = line.split(":")
            if len(split) != 2: continue
            d[split[0]] = split[1].replace("\n", "").replace(" ", "")
    return d

def scrap_outputs():
    """ Scrap configurations and validation results for all directory in
    output directory and write in pandas dataframe. """
    out_path = os.environ["SR_PROJECT_OUTS_PATH"]
    out_dirs = os.listdir(out_path)
    out_dirs = [os.path.join(out_path, x) for x in out_dirs]
    out_dirs = [x for x in out_dirs if os.path.isdir(x)]
    # Iterate over all output directories, search for configuration and
    # validation file and add to global dataframe.
    config_keys  = ["type", "format", "external", "no_augment", "model", "loss",
                    "data_train", "scales_train"]
    results_keys = ["scale", "dataset", "RUNTIME_UP", "RUNTIME_DW", "RUNTIME_AL",
                    "PSNR_SLR_mean", "PSNR_SLR_best", "PSNR_SHRT_mean",
                    "PSNR_SHRT_best", "PSNR_SHRB_mean", "PSNR_SHRB_best",
                    "PSNR_SCOLT_best", "PSNR_SCOLT_mean", "PSNR_SGRY_best",
                    "PSNR_SGRY_mean"]
    overall_keys = config_keys + results_keys
    scrapped = {x: [] for x in overall_keys}
    for dir in out_dirs:
        if not os.path.isfile(os.path.join(dir, "config.txt")): continue
        config  = read_config_file(os.path.join(dir, "config.txt"))
        if not all([x in config.keys() for x in config_keys]): continue
        if not os.path.isfile(os.path.join(dir, "validations.csv")): continue
        results_df = pd.read_csv(os.path.join(dir, "validations.csv"))
        for index, row in results_df.iterrows():
            for x in scrapped.keys(): scrapped[x].append(None)
            for x in config_keys: scrapped[x][-1] = config[x]
            for x in results_keys:
                if x not in row.keys(): continue
                else: scrapped[x][-1] = row[x]
    return pd.DataFrame.from_dict(scrapped)
