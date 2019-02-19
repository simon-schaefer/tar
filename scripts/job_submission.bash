#!/bin/bash

SCRIPT="test.py"

# Checking whether environment is set. 
if [ -z "$REMOTE_IS_SET" ]; then
    echo "Environment variables not set, source 'source_env.bash' !"
    exit 0
fi
# Print header. 
source "$LOCAL_DIR_SCRIPTS/header.bash"

qsub $LOCAL_DIR_SCRIPTS/job_script.bash $REMOTE_PROJECT_HOME $REMOTE_VIRTUAL_ENV_PATH $SCRIPT