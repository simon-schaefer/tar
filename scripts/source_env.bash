#!/bin/bash

REAL_PATH="$(cd "$(dirname "$BASH_SOURCE")"; pwd)/$(basename "$BASH_SOURCE")"
REAL_PATH=`dirname "$REAL_PATH"`
source "$REAL_PATH/parameters.bash"
source "$SR_PROJECT_SCRIPTS_PATH/header.bash"

# Setting for BIWI clusters. 
source /home/sgeadmin/BIWICELL/common/settings.sh

# Local language settings (suppressing local language error). 
if [ ! -z "$LANGUAGE" ]
then
    export LANGUAGE=en_US.UTF-8
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    locale-gen en_US.UTF-8
fi

# Update github repository. 
cd $SR_PROJECT_PROJECT_HOME/
git fetch
git pull --rebase

# Install self-python-package. 
cd $SR_PROJECT_PROJECT_HOME
pip install .

# Set environment set flag. 
export SR_PROJECT_IS_SET="True"
