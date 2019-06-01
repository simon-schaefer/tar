#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Utility functions for evaluations.
# =============================================================================
import importlib
import os
import pandas as pd
import tar
import torch

def read_config_file(fname):
    d = {}
    with open(fname) as f:
        for line in f:
            split = line.split(":")
            if len(split) != 2: continue
            d[split[0]] = split[1].replace("\n", "").replace(" ", "")
    return d

def num_model_params(model_name):
    module = importlib.import_module('tar.models.' + args.model.lower())
    model  = module.build_net().to(self.device)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_logging(log_file_path):
    """ Parse information from log.txt file and return in dictionary.
    Supported information: epoch, overall loss. """
    assert os.path.isfile(log_file_path)
    num_lines    = sum(1 for line in open(log_file_path, 'r'))
    logging_list = []
    epoch_dict   = {"epoch":0, "loss": 0}
    print("\nRead logging file {} ... ".format(log_file_path))
    with open(log_file_path, "r") as file:
        for il, line in enumerate(file):
            if line.count("Epoch") > 0:
                m = filter_string(line.split("\t")[0]).split(" ")[1]
                epoch_dict["epoch"] = int(m)
            if line.count("TOTAL") > 0:
                m = filter_string(line.split("TOTAL")[1].split("]")[0],space=True)
                epoch_dict["loss"] = float(m)
            if line.count("Validation Total") > 0:
                logging_list.append(epoch_dict)
                epoch_dict   = {"epoch":0, "loss": 0}
    epoch_df = pd.DataFrame(logging_list)
    return epoch_df

def scrap_outputs(directory):
    """ Scrap configurations and validation results for all directory in
    given directory and write in pandas dataframe. """
    out_path = directory
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
        #if not all([x in config.keys() for x in config_keys]): continue
        if not os.path.isfile(os.path.join(dir, "validations.csv")): continue
        results_df = pd.read_csv(os.path.join(dir, "validations.csv"))
        for index, row in results_df.iterrows():
            for x in scrapped.keys(): scrapped[x].append(None)
            for x in config_keys:
                if x in config: scrapped[x][-1] = config[x]
            for x in results_keys:
                if x == "RUNTIME_AL" and "RUNTIME" in row.keys():
                    scrapped[x][-1] = row["RUNTIME"]
                elif x not in row.keys(): continue
                else: scrapped[x][-1] = row[x]
    return pd.DataFrame.from_dict(scrapped)

def filter_string(phrase, punctuation='[^!?]+:', space=False):
    for x in phrase.lower():
        if x in punctuation:
            phrase = phrase.replace(x, "")
    if space: phrase = phrase.replace(" ", "")
    return phrase

# =============================================================================
# Created By  : Simon Schaefer
# Description : Network architecture visualization.
# =============================================================================
# import torch
# from torch.autograd import Variable
# import tar.models as tar_models
# import tar.models.aetad
#
# model = tar_models.aetad.AETAD()
# dummy_input = Variable(torch.randn(4, 3, 64, 64))
# torch.onnx.export(model, dummy_input, "aetad.onnx")
