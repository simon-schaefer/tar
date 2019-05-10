#!/bin/bash

# Login to cluster environment.
qrsh -l gpu=1 -l h_vmem=20G -q gpu.24h.q@*
qrsh -l gpu=0 -q "*@biwirenderXX"

# Check gpu and user.
grep -h $(whoami) /tmp/lock-gpu*/info.txt
nvidia-smi

# Job submission.
qsub scripts/jobs/job24.bash
qsub scripts/jobs/job48.bash

# Queue handling.
qstat
qdel -u sischaef
qquota -u sischaef | grep max_gpu_per_user

# Background calculation (https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/).
screen              # start session (CRTL+A+D to detach).
screen -ls          # list running sessions.
screen -r "id"      # reattach session.
