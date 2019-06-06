#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Compare psnr and runtime of models.
# Arguments   : Filter (format="key1:value1&key2:value2...")
# =============================================================================
import argparse
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit

from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="psnr_time")
parser.add_argument("--directory", type=str, default="")
parser.add_argument("--psnr_tag", type=str, default="SHRT_mean",
                    choices=("SHRT_mean", "SCOLT_mean",
                             "SHRT_best", "SCOLT_best"))
parser.add_argument("--filter", type=str, default="")
parser.add_argument("--scaling", type=float, default=0.0)
args = parser.parse_args()

print("Scrapping outs data ...")
psnr_tag = "PSNR_{}".format(args.psnr_tag)
dir = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], args.directory)
data = scrap_outputs(directory=dir)

print("... add baseline results")
data = add_baseline_results(data, scaling=args.scaling)

print("... filtering outs data")
for key_value in args.filter.split("&"):
    if key_value == "": continue
    key, value = key_value.split(":")
    if len(value.split("/")) > 0:
        data = data[data[key].isin(value.split("/"))]
    else:
        data = data[data[key] == value]

print("... averaging over models")
data = average_key_over_key(data, psnr_tag, "model", "dataset", args.scaling)

print("... plotting psnr boxplot plots")
f, axes = plt.subplots(figsize=(8,8))
sns.violinplot(x="model",y=psnr_tag,data=data, orient='v')
plt.ylabel("PSNR [dB]")
plt.xticks(rotation=20)
plt.savefig(save_path("pnsr_boxplot.png"))
plt.close()

# print("... regressing non-linear function")
# def func(x, a, b, c, d):
#     return a + b*x + c*np.exp(d*x)

print("... plotting complexity-psnr-correlation plot")
unique_datasets  = np.unique(data["dataset"])
num_unique_dsets = len(unique_datasets)
f, axes = plt.subplots(1, num_unique_dsets, figsize=(8*num_unique_dsets,8))
for id, dataset in enumerate(unique_datasets):
    xs, ys = [], []
    # Plot actual data.
    for index, row in data.iterrows():
        if not row["dataset"] == dataset: continue
        x, y = row["complexity"], row["{}_model_dataset_avg".format(psnr_tag)]
        axes[id].scatter(x, y, marker='x')
        axes[id].text(x+.03, y+.03, row["model"], fontsize=9)
        xs.append(x/100000); ys.append(y)
    # Plot non-linear regression.
    # guess_params = [120, 100, -10, -0.01]
    # popt, _      = curve_fit(func, xs, ys, guess_params)
    # xs_plot = (np.linspace(np.min(xs), np.max(xs), num=20)*100000).tolist()
    # print(popt)
    # print("-"*20)
    # ys_plot = [func(x/100000, *popt) for x in xs_plot]
    # axes[id].plot(xs_plot, ys_plot, '--')
    axes[id].set_title(dataset)
    axes[id].set_ylabel("PSNR [dB]")
    axes[id].set_xlabel("Model complexity")
plt.savefig(save_path("psnr_complexity_linear.png"))
plt.close()
