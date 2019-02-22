#!/bin/bash

# Source environment (create env. variables).
REAL_PATH="$(cd "$(dirname "$BASH_SOURCE")"; pwd)/$(basename "$BASH_SOURCE")"
REAL_PATH=`dirname "$REAL_PATH"`
source "$REAL_PATH/parameters.bash"
source "$SR_PROJECT_SCRIPTS_PATH/header.bash"

# Setting for BIWI clusters. 
echo $'\nBuilding BIWI cluster environment ...'
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
echo "Updating GitHub repository ..."
cd $SR_PROJECT_PROJECT_HOME/
git fetch
git pull --rebase

# Login to virtual environment.
echo "Sourcing virtual environment ..."
source $SR_PROJECT_VIRTUAL_ENV_PATH/bin/activate

# Install self-python-package. 
echo "Installing package ..."
cd $SR_PROJECT_PROJECT_HOME
pip install .

# Set environment set flag. 
export SR_PROJECT_IS_SET="True"
echo "Successfully built environment !"

# Download datasets.  
echo $'\nDownloading datasets ...'
cd $SR_PROJECT_HOME
if [ ! -d "data" ]; then
    mkdir data
fi
cd data
## DIV2K dataset.
echo "DIV2K dataset ..."
if [ ! -d "DIV2K_train_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    unzip DIV2K_train_HR.zip
    rm DIV2K_train_HR.zip
fi
if [ ! -d "DIV2K_valid_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
fi
# MNIST dataset. 
echo "MNIST dataset ..."
if [ ! -d "MNIST_train" ]; then
    wget https://pjreddie.com/media/files/mnist_train.tar.gz
    tar -xvf mnist_train.tar.gz
    rm mnist_train.tar.gz
    mv "train" "MNIST_train"
fi
if [ ! -d "MNIST_valid" ]; then
    wget https://pjreddie.com/media/files/mnist_test.tar.gz
    tar -xvf mnist_test.tar.gz
    rm mnist_test.tar.gz
    mv "test" "MNIST_valid"
fi

cd $SR_PROJECT_PROJECT_HOME
echo $'\nSuccessfully set up project !'