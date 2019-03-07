#!/bin/bash

# Login to cluster environment. 
qrsh -l gpu -l h_vmem=20G -q gpu.middle.q@*

# Check gpu and user. 
grep -h $(whoami) /tmp/lock-gpu*/info.txt

# Queue information. 
qstat

# Background calculation. 
screen

### Example Jobs ####
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TAD_MNIST --no_augment
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TAD_MNIST --data_range "1-20/21-25" --no_augment
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TEST --no_augment
