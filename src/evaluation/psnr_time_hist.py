#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# =============================================================================
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

fdata = "psnr_time_network.csv"
data = pd.read_csv(fdata)
data = data[data.Type != "Runtime"]
print(data.head())
fig = plt.figure()
sns.catplot(x="Type", y="Value", hue="Network",
            data=data, kind="bar", height=6, aspect=1.5)
plt.ylabel("PSNR")
plt.ylim([25,35])
plt.savefig(fdata.replace(".csv", ".png"))
