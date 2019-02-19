#!/bin/bash

source "parameters.bash"
source "$REMOTE_SCRIPTS_PATH/header.bash"

# Setting for BIWI clusters. 
source /home/sgeadmin/BIWICELL/common/settings.sh

# Local language settings (suppressing local language error). 
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8

# Update github repository. 
cd $REMOTE_PROJECT_HOME
git fetch
git pull --rebase

# Set environment set flag. 
export REMOTE_IS_SET="True"
