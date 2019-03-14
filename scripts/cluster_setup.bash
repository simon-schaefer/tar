#!/bin/bash

#Cuda:0 is always the first visible GPU. So if you set CUDA_VISIBLE_DEVICES 
# to another index (e.g. 1), this GPU is referred to as cuda:0. 
# Alternatively you could specify the device as torch.device('cpu') for 
# running your model/tensor on CPU.
# The global GPU index (which is necessary to set CUDA_VISIBLE_DEVICES in the 
# right way) can be seen by the nvidia-smi command in the shell/bash.
export CUDA_VISIBLE_DEVICES=1