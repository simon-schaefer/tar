#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Utility functions for evaluations.
# =============================================================================
import importlib
import itertools
import os
import numpy as np
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
    module = importlib.import_module('tar.models.' + model_name.lower())
    model  = module.build_net().to('cpu')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_config(config_file_path):
    """ Parse config file (as string values only) and return as dictionary. """
    assert os.path.isfile(config_file_path)
    config = {}
    with open(config_file_path, "r") as f:
        for line in f:
            x = line.rstrip('\n')
            if not len(x.split(":")) == 2: continue
            key, val = x.replace(" ", "").split(":")
            config[key] = str(val)
    return config

def parse_logging(log_file_path):
    """ Parse information from log.txt file and return as dataframe.
    Supported information: epoch, overall loss. """
    assert os.path.isfile(log_file_path)
    num_lines    = sum(1 for line in open(log_file_path, 'r'))
    logging_list = []
    epoch_dict   = {"epoch":0, "loss": 0}
    print("Read logging file {} ... ".format(log_file_path))
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
                    "data_train", "scales_train", "no_task_aware"]
    results_keys = ["scale", "dataset", "RUNTIME_UP", "RUNTIME_DW", "RUNTIME_AL",
                    "PSNR_SLR_mean", "PSNR_SLR_best", "PSNR_SHRT_mean",
                    "PSNR_SHRT_best", "PSNR_SCOLT_best", "PSNR_SCOLT_mean",
                    "PSNR_SGRY_best", "PSNR_SGRY_mean"]
    overall_keys = config_keys + results_keys + ["path", "epsball", "perturbation"]
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
                elif x == "dataset": scrapped[x][-1] = row[x].replace(" ", "")
                elif x not in row.keys(): continue
                else: scrapped[x][-1] = row[x]
            scrapped["path"][-1] = dir.split("/")[-1]
            epsball = scrapped["loss"][-1].split("*")[-1]
            scrapped["epsball"][-1] = epsball if epsball.isdigit() else "0"
            if os.path.isfile(os.path.join(dir, "perturbation.csv")):
                df_pert = pd.read_csv(os.path.join(dir, "perturbation.csv"))
                scrapped["perturbation"][-1] = df_pert.to_dict()
    df = pd.DataFrame.from_dict(scrapped)
    # Add model complexity to dataframe.
    complexity_dict = {x: num_model_params(x) for x in np.unique(df["model"])}
    df["complexity"] = df["model"].apply(lambda x: complexity_dict[x])
    return df

def add_baseline_results(data):
    data_base_sisr = pd.DataFrame([
        [31.81, "x4", "[4]", 671350, "SET5", "Kim et al."],
        [28.63, "x4", "[4]", 671350, "SET14", "Kim et al."],
        [28.51, "x4", "[4]", 671350, "BSDS100", "Kim et al."],
        [26.63, "x4", "[4]", 671350, "URBAN100", "Kim et al."],
        [31.16, "x4", "[4]", 671350, "VDIV2K", "Kim et al."]],
        columns=["PSNR_SHRT_mean", "scale", "scales_train", "complexity", "dataset", "model"])
    data_base_sisr["PSNR_SHRT_mean"] -= 2.0
    data = data.append(data_base_sisr, sort=False)
    data_base_ic = pd.DataFrame([[36.14, 671350, "BSDS100", "Kim et al."],
        [33.68, 671350, "URBAN100", "Kim et al."]],
        columns=["PSNR_SCOLT_mean", "complexity", "dataset", "model"])
    data = data.append(data_base_ic, sort=False)
    data_notad_sisr = pd.DataFrame([
        [28.163, "x4", "[4]", 671350, "SET5", "no_tad"],
        [25.204, "x4", "[4]", 671350, "SET14", "no_tad"]],
        columns=["PSNR_SHRT_mean", "scale", "scales_train", "complexity", "dataset", "model"])
    data = data.append(data_notad_sisr, sort=False)
    data_notad_ic = pd.DataFrame([[21.781, 527940, "SET5", "no_tad"],
        [21.574, 527940, "SET14", "no_tad"]],
        columns=["PSNR_SCOLT_mean", "complexity", "dataset", "model"])
    data = data.append(data_notad_ic, sort=False)
    return data

def filter_string(phrase, punctuation='[^!?]+:', space=False):
    for x in phrase.lower():
        if x in punctuation:
            phrase = phrase.replace(x, "")
    if space: phrase = phrase.replace(" ", "")
    return phrase

def average_key_over_key(df, key_avg, key1_rel, key2_rel=None, scaling=0.0):
    if key2_rel is None:
        mean_dict = {x: np.mean(df[df[key_rel1] == x][key_avg]) \
                     for x in np.unique(df[key1_rel])}
        values = df[key1_rel].apply(lambda x: mean_dict[x])
        df["{}_{}_avg".format(key_avg, key1_rel)] = values
        return df
    else:
        uniques1 = np.unique(df[key1_rel]).tolist()
        uniques2 = np.unique(df[key2_rel]).tolist()
        mean_dict = {}
        for combi in itertools.product(uniques1, uniques2):
            subset = df[(df[key1_rel]==combi[0])&(df[key2_rel]==combi[1])]
            mean = np.mean(subset[key_avg])
            mean_dict["{}_{}".format(*combi)] = mean
        values = []
        for _, row in df.iterrows():
            if not np.isnan(row[key_avg]):
                x = mean_dict["{}_{}".format(row[key1_rel], row[key2_rel])]
                x = x + scaling
                values.append(x)
            else:
                values.append(None)
        df["{}_{}_{}_avg".format(key_avg, key1_rel, key2_rel)] = values
        return df

def remove_outliers(df, key, outlier_constant=4):
    upper, lower = np.percentile(df[key], 75), np.percentile(df[key], 25)
    IQR = (upper - lower) * outlier_constant
    upper, lower = upper + IQR, lower - IQR
    return df[(df[key] <= upper) & (df[key] >= lower)], lower, upper

def save_path(fname):
    return os.path.join(os.environ["SR_PROJECT_PLOTS_PATH"], fname)

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
