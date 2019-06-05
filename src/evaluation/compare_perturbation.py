#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Load and compare perturbation curves from output directory.
# Arguments   : Out directiory paths (seperated by &).
#               Plot labels (tags) (seperated by &).
# =============================================================================
import argparse
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils import save_path, scrap_outputs

# Parse out directories from user input.
parser = argparse.ArgumentParser(description="perturbation_curves")
parser.add_argument("--directory", type=str, default="")
args = parser.parse_args()

print("Scrapping and filtering outs data ...")
dir = os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], args.directory)
data = scrap_outputs(directory=dir)
epsilons, eps_psnrs = [], {}
for index, row in data.iterrows():
    if row["perturbation"] is None: continue
    eps_psnrs[row["epsball"]] = list(row["perturbation"]["SET5x4"].values())
    epsilons = list(row["perturbation"]["epsilon"].values())

print("... plotting epsball-psnr-correlation plot")
f, axes = plt.subplots()
for eps in eps_psnrs.keys():
    plt.plot(epsilons, eps_psnrs[eps], label="epsilon={}".format(eps))
plt.legend()
plt.ylabel("PSNR [dB]")
plt.xlabel("Perturbation - Image = [0,1]")
plt.savefig(save_path("epsball_psnr.png"))
plt.close()

print("... plotting epsball-msr-correlation plot")
f, axes = plt.subplots()
for eps in eps_psnrs.keys():
    mse = np.power(np.divide(eps_psnrs[eps], 20), -10)
    plt.plot(epsilons, mse, label="epsilon={}".format(eps))
plt.ylim([0, 4000])
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Perturbation - Image = [0,1]")
plt.savefig(save_path("epsball_mse.png"))
plt.close()
