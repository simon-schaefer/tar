#!/bin/bash

# Login to cluster environment. 
qrsh -l gpu=1 -l h_vmem=20G -q gpu.middle.q@*
qrsh -l gpu=0 -q "*@biwirenderXX"

# Check gpu and user. 
grep -h $(whoami) /tmp/lock-gpu*/info.txt
nvidia-smi

# Queue information. 
qstat

# Background calculation (https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/). 
screen              # start session (CRTL+A+D to detach). 
screen -ls          # list running sessions. 
screen -r "id"      # reattach session. 

### Example Jobs ####
# src/super_resolution/main.py --template IM_AE_TAD_DIV2K --no_augment --verbose --data_range "3-4/3-4" --batch_size 1 --loss "HR*100*L1+LR*1*L1" --lr 1e-4
# src/super_resolution/main.py --template IM_AE_TAD_DIV2K_SMALL --no_augment --verbose --print_every 10
# src/super_resolution/main.py --template IM_AE_TAD_DIV2K --no_augment --verbose --print_every 10
# src/super_resolution/main.py --template IM_AE_TAD_MNIST --no_augment
# src/super_resolution/main.py --template IM_AE_TAD_MNIST --data_range "1-20/21-25" --no_augment --verbose
# src/super_resolution/main.py --template IM_AE_TEST --no_augment
