#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Apply model and measure time (works without setup !).
#               Hard-Coded fast test.
# =============================================================================
import imageio
import importlib
import numpy as np
import pandas as pd
import time
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tar

device     = "cpu"
model_name = "AETAD"
model_path = "models/iscale4/model"
img_path   = "src/tests/ressources/0002.png"
num_trials = 10

# Load model.
module = importlib.import_module('tar.models.' + model_name.lower())
model  = module.build_net().to(device)
load_from = None
kwargs = {'map_location': lambda storage, loc: storage}
path = model_path + "/model_latest.pt"
load_from = torch.load(path, **kwargs)
model.load_state_dict(load_from, strict=False)

# Load image.
img = imageio.imread(img_path)

# Apply model and measure time.
print("Applying on Mac Pro 2015 cpu ...")
runtimes = []
for scale in [2,4]:
    print("... scale factor {}".format(scale))
    for i in range(num_trials):
        print("... trial {}".format(i+1))
        start_time = time.time()
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        np_transpose = np.expand_dims(np_transpose, axis=0)
        hr = torch.from_numpy(np_transpose).float()
        x = model.encode(hr.clone())
        if scale == 4:
            x = model.encode(x)
            x = model.decode(x)
        x = model.decode(x)
        runtimes.append({"runtime": time.time() - start_time, "scale": scale})
df = pd.DataFrame(runtimes)

# Plotting distribution of runtimes.
print("... plotting runtimes as boxplot")
sns.boxplot(x="scale", y="runtime", data=df)
plt.ylabel("runtime [s]")
plt.savefig("mac_runtimes.png")
plt.close()
