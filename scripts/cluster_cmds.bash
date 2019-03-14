#!/bin/bash

# Login to cluster environment. 
qrsh -l gpu -l h_vmem=20G -q gpu.middle.q@*

# Check gpu and user. 
grep -h $(whoami) /tmp/lock-gpu*/info.txt

# Queue information. 
qstat

# Background calculation (https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/). 
screen              # start session (CRTL+A+D to detach). 
screen -ls          # list running sessions. 
screen -r "id"      # reattach session. 

### Example Jobs ####
# src/super_resolution/main.py --template IM_AE_TAD_DIV2K --no_augment --verbose --print_every 10
# src/super_resolution/main.py --template IM_AE_TAD_MNIST --no_augment
# src/super_resolution/main.py --template IM_AE_TAD_MNIST --data_range "1-20/21-25" --no_augment
# src/super_resolution/main.py --template IM_AE_TEST --no_augment
