import multiprocessing
import os
import subprocess
import sys

import torch

def mp_worker(gpu):
    print(torch.cuda.get_device_properties(gpu))

# Print CUDA versions information. 
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
subprocess.call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# Print GPU information. 
gpus = list(range(torch.cuda.device_count()))
processes = [multiprocessing.Process(target=mp_worker, args=(gpui,)) for gpui in gpus]
for process in processes:
    process.start()
for process in processes:
    process.join()