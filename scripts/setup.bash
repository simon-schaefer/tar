#!/bin/bash

# Source environment (create env. variables).
REAL_PATH="$(cd "$(dirname "$BASH_SOURCE")"; pwd)/$(basename "$BASH_SOURCE")"
REAL_PATH=`dirname "$REAL_PATH"`
source "$REAL_PATH/source_env.bash"

# Download datasets.  
cd $SR_PROJECT_PROJECT_HOME
mkdir data
cd data
## DIV2K dataset. 
if [ ! -d "DIV2K_train_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    unzip DIV2K_train_HR.zip
    rm DIV2K_train_HR.zip
fi
if [ ! -d "DIV2K_train_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
fi
cd $SR_PROJECT_PROJECT_HOME