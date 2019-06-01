#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# =============================================================================
from utils import scrap_outputs

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

data = scrap_outputs()
data["RUNTIME_AL"] = round(data["RUNTIME_AL"]*1000, 2)
data = data[data.type == "SCALING"]
data = data[(data.model == "AETAD_SMALL") | (data.model == "AETAD_LARGE")]

f, axes = plt.subplots(1, 2, figsize=(10,5))
sns.boxplot(x="model",y="PSNR_SHRT_best",data=data, orient='v', ax=axes[0])
axes[0].set_ylabel("PSNR [dB]")
sns.boxplot(x="model",y="RUNTIME_AL", data=data, orient='v', ax=axes[1])
axes[1].set_ylabel("RUNTIME OVERALL [ms]")
plt.savefig("psnr_time_hist.png")
plt.close()
