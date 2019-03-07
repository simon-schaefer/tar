#!/bin/bash

export SR_PROJECT_NAME="super_resolution"

export SR_PROJECT_HOME="/scratch_net/biwidl211/sischaef"
export SR_PROJECT_PROJECT_HOME="$SR_PROJECT_HOME/$SR_PROJECT_NAME"
export SR_PROJECT_SCRIPTS_PATH="$SR_PROJECT_PROJECT_HOME/scripts"
export SR_PROJECT_DATA_PATH="$SR_PROJECT_HOME/data"
export SR_PROJECT_OUTS_PATH="$SR_PROJECT_HOME/outs"
export SR_PROJECT_VIRTUAL_ENV_PATH="$SR_PROJECT_HOME/venv"

# Source environment (create env. variables).
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
cd $SR_PROJECT_HOME
if [ ! -d "venv" ]; then
    echo "Creating virtual environment ..."
    mkdir venv
    virtualenv -p python3 venv
fi
echo "Sourcing virtual environment ..."
source $SR_PROJECT_VIRTUAL_ENV_PATH/bin/activate

# Install self-python-package. 
echo "Installing package ..."
cd $SR_PROJECT_PROJECT_HOME
pip install -r requirements.txt
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
if [ ! -d "DIV2K" ]; then
    mkdir DIV2K
fi 
cd DIV2K
if [ ! -d "DIV2K_train_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    unzip DIV2K_train_HR.zip
    rm DIV2K_train_HR.zip
fi
if [ ! -d "DIV2K_train_LR_bicubic" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
    unzip DIV2K_train_LR_bicubic_X2.zip
    rm DIV2K_train_LR_bicubic_X2.zip
    mv DIV2K_train_LR_bicubic/X2/* DIV2K_train_LR_bicubic/
    rm -r DIV2K_train_LR_bicubic/X2/

fi
if [ ! -d "DIV2K_valid_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
    cp DIV2K_valid_HR/* DIV2K_train_HR/
fi
if [ ! -d "DIV2K_valid_LR_bicubic" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
    unzip DIV2K_valid_LR_bicubic_X2.zip
    rm DIV2K_valid_LR_bicubic_X2.zip
    mv DIV2K_valid_LR_bicubic/X2/* DIV2K_valid_LR_bicubic/
    rm -r DIV2K_valid_LR_bicubic/X2/
    cp DIV2K_valid_LR_bicubic/* DIV2K_train_LR_bicubic/
fi
cd $SR_PROJECT_DATA_PATH
# MNIST dataset. 
echo "MNIST dataset ..."
if [ ! -d "MNIST" ]; then
    mkdir MNIST
fi
cd MNIST
if [ ! -d "MNIST_train_HR" ]; then
    wget https://pjreddie.com/media/files/mnist_train.tar.gz
    tar -xvf mnist_train.tar.gz
    rm mnist_train.tar.gz
    mv "train" "MNIST_train_HR"
fi
if [ ! -d "MNIST_valid_HR" ]; then
    wget https://pjreddie.com/media/files/mnist_test.tar.gz
    tar -xvf mnist_test.tar.gz
    rm mnist_test.tar.gz
    mv "test" "MNIST_valid_HR"
    cp MNIST_valid_HR/* MNIST_train_HR/
fi
if [ ! -d "MNIST_train_LR_bicubic" ]; then
    python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/MNIST/MNIST_train_HR 2 False
fi
cd $SR_PROJECT_DATA_PATH

# Build outs directory.
echo $'\nBuilding output directory ...'
cd $SR_PROJECT_HOME
if [ ! -d "outs" ]; then
    mkdir outs
fi

cd $SR_PROJECT_PROJECT_HOME
echo $'\nSuccessfully set up project !'