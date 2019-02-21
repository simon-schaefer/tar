#!/bin/bash

# Checking whether environment is set. 
if [ -z "$SR_PROJECT_IS_SET" ]; then
    echo "Environment variables not set, source 'source_env.bash' !"
    REAL_PATH="$(cd "$(dirname "$BASH_SOURCE")"; pwd)/$(basename "$BASH_SOURCE")"
    REAL_PATH=`dirname "$REAL_PATH"`
    source "$REAL_PATH/source_env.bash"
fi

# Activate virtual environment. 
source $SR_PROJECT_VIRTUAL_ENV_PATH/bin/activate
