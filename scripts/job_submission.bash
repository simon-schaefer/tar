#!/bin/bash

SCRIPT_AND_ARGS="$1"

qsub $SR_PROJECT_SCRIPTS_PATH/job_script.bash $SR_PROJECT_PROJECT_HOME $SR_PROJECT_VIRTUAL_ENV_PATH $SCRIPT_AND_ARGS

### Examples ####
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TAD_MNIST --no_augment
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TAD_MNIST --data_range "1-20/21-25" --no_augment
# bash scripts/job_submission.bash src/super_resolution/main.py --template IM_AE_TEST --no_augment