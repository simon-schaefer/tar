#!/bin/bash

SCRIPT="test.py"

# Checking whether environment is set. 
if [ -z "$SR_PROJECT_IS_SET" ]; then
    echo "Environment variables not set, source 'source_env.bash' !"
    exit 0
fi
# Print header. 
source "$LOCAL_DIR_SCRIPTS/header.bash"

qsub $LOCAL_DIR_SCRIPTS/job_script.bash $SR_PROJECT_PROJECT_HOME $SR_PROJECT_VIRTUAL_ENV_PATH $SCRIPT