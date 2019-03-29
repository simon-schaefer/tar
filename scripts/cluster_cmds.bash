#!/bin/bash

# Login to cluster environment. 
qrsh -l gpu=1 -l h_vmem=20G -q gpu.middle.q@*
qrsh -l gpu=0 -q "*@biwirenderXX"

# Check gpu and user. 
grep -h $(whoami) /tmp/lock-gpu*/info.txt
nvidia-smi

# Job submission. 
qsub scripts/job.bash 

# Queue handling. 
qstat 
qdel -u sischaef
qquota -u sischaef | grep max_gpu_per_user

# Background calculation (https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/). 
screen              # start session (CRTL+A+D to detach). 
screen -ls          # list running sessions. 
screen -r "id"      # reattach session. 

### Example Jobs ####
# --template=IM_AE_TAD_DIV2K --no_augment --verbose --data_range=1-700/1-10 --batch_size=1 --loss=HR*100*L1+LR*1*L1 --lr=1e-4
# --template=IM_AE_TAD_DIV2K --no_augment --verbose --data_range=3-4/3-4 --batch_size=1 --loss=HR*100*L1+LR*1*L1 --epochs 4 --data_valid="SET14"
# --template=IM_AE_TEST --no_augment