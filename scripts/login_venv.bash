#!/bin/bash

# Checking whether environment is set. 
if [ -z "$REMOTE_IS_SET" ]; then
    echo "Environment variables not set, source 'source_env.bash' !"
    exit 0
fi

# Activate virtual environment. 
source $REMOTE_VIRTUAL_ENV_PATH/bin/activate
