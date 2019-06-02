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

from utils import average_key_over_key, remove_outliers, scrap_outputs

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="psnr_time")
parser.add_argument("--filter", type=str, default="")
args = parser.parse_args()

print("Scrapping and filtering outs data ...")
data = scrap_outputs(directory=os.environ["SR_PROJECT_OUTS_PATH"])
data["RUNTIME_AL"] = round(data["RUNTIME_AL"]*1000, 2)
for key_value in args.filter.split("&"):
    if key_value == "": continue
    key, value = key_value.split(":")
    if len(value.split("/")) > 0:
        data = data[data[key].isin(value.split("/"))]
    else:
        data = data[data[key] == value]
print("... averaging over models")
data = average_key_over_key(data, "PSNR_SHRT_mean", "model", "dataset")
data = average_key_over_key(data, "PSNR_SHRT_best", "model", "dataset")
#data, lower, upper = remove_outliers(data, "RUNTIME_AL")
#print("... removing runtime outliers outside of [{},{}]".format(lower, upper))
data.to_csv("data_filtered.csv")
data_sorted = data.sort("RUNTIME_AL")
data_sorted.to_csv("data_sorted.csv")

print("... plotting psnr boxplot plots")
f, axes = plt.subplots(figsize=(8,8))
sns.boxplot(x="model",y="PSNR_SHRT_mean",data=data, orient='v')
plt.ylabel("PSNR [dB]")
plt.xticks(rotation=30)
plt.savefig("pnsr_boxplot.png")
plt.close()

print("... plotting runtime boxplot plots")
f, axes = plt.subplots(figsize=(8,8))
sns.boxplot(x="model",y="RUNTIME_AL", data=data, orient='v')
plt.ylabel("RUNTIME OVERALL [ms]")
plt.xticks(rotation=30)
plt.savefig("runtime_boxplot.png")
plt.close()

print("... plotting complexity-psnr-correlation plot")
f, axes = plt.subplots(figsize=(8,8))
sns.catplot(x="complexity", y="PSNR_SHRT_best_model_dataset_avg",
            hue="model", col="dataset", data=data)
axes.set_ylabel("PSNR [dB]")
axes.set_xlabel("Model complexity")
plt.savefig("psnr_complexity_linear.png")
plt.close()
