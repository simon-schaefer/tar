#!/bin/bash

SCRIPT_AND_ARGS="$1"

qsub $SR_PROJECT_SCRIPTS_PATH/job_script.bash $SR_PROJECT_PROJECT_HOME $SR_PROJECT_VIRTUAL_ENV_PATH $SCRIPT_AND_ARGS

### Examples ####
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_SR_TAD --data_train MNIST --data_test MNIST --n_colors 1 --patch_size 10
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TEST --no_augment