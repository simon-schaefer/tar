#!/bin/bash

export SR_PROJECT_NAME="tar"

export SR_PROJECT_HOME="/scratch_net/biwidl215/sischaef"
export SR_PROJECT_PROJECT_HOME="$SR_PROJECT_HOME/$SR_PROJECT_NAME"
export SR_PROJECT_SCRIPTS_PATH="$SR_PROJECT_PROJECT_HOME/scripts"
export SR_PROJECT_DATA_PATH="$SR_PROJECT_HOME/data"
export SR_PROJECT_OUTS_PATH="$SR_PROJECT_HOME/outs"
export SR_PROJECT_MODELS_PATH="$SR_PROJECT_PROJECT_HOME/models"
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
echo $'\nUpdating GitHub repository ...'
cd $SR_PROJECT_PROJECT_HOME/
#git stash -a
git fetch
git pull --rebase
git status

# Login to virtual environment.
echo $'\nSourcing virtual environment ...'
cd $SR_PROJECT_HOME
if [ ! -d "venv" ]; then
    echo "Creating virtual environment ..."
    mkdir venv
    virtualenv -p python3 venv
fi
source $SR_PROJECT_VIRTUAL_ENV_PATH/bin/activate

# Install self-python-package. 
echo $'\nInstalling package ...'
cd $SR_PROJECT_PROJECT_HOME
pip install -r requirements.txt --user
pip install -e . --user

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
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
    unzip DIV2K_train_LR_bicubic_X4.zip
    rm DIV2K_train_LR_bicubic_X4.zip
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
    unzip DIV2K_train_LR_x8.zip
    rm DIV2K_train_LR_x8.zip
    mv DIV2K_train_LR_x8 DIV2K_train_LR_bicubic/X8
fi
if [ ! -d "DIV2K_valid_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
fi
if [ ! -d "DIV2K_valid_LR_bicubic" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
    unzip DIV2K_valid_LR_bicubic_X2.zip
    rm DIV2K_valid_LR_bicubic_X2.zip
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
    unzip DIV2K_valid_LR_bicubic_X4.zip
    rm DIV2K_valid_LR_bicubic_X4.zip
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip
    unzip DIV2K_valid_LR_x8.zip
    rm DIV2K_valid_LR_x8.zip
    mv DIV2K_valid_LR_x8 DIV2K_valid_LR_bicubic/X8
fi

# MNIST dataset. 
# echo "MNIST dataset ..."
# if [ ! -d "MNIST" ]; then
#     mkdir MNIST
# fi
# cd MNIST
# if [ ! -d "MNIST_train_HR" ]; then
#     wget https://pjreddie.com/media/files/mnist_train.tar.gz
#     tar -xvf mnist_train.tar.gz
#     rm mnist_train.tar.gz
#     mv "train" "MNIST_train_HR"
# fi
# if [ ! -d "MNIST_valid_HR" ]; then
#     wget https://pjreddie.com/media/files/mnist_test.tar.gz
#     tar -xvf mnist_test.tar.gz
#     rm mnist_test.tar.gz
#     mv "test" "MNIST_valid_HR"
#     cp MNIST_valid_HR/* MNIST_train_HR/
# fi
# if [ ! -d "MNIST_train_LR_bicubic" ]; then
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/MNIST/MNIST_train_HR 2
# fi

# SIMPLE dataset. 
# echo "SIMPLE dataset ..."
# cd $SR_PROJECT_DATA_PATH
# if [ ! -d "SIMPLE" ]; then
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/create_simple_dataset.py
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/SIMPLE/HR 2
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/SIMPLE/HR 4
# fi

# Validation datasets. 
echo "Validation datasets ..."
cd $SR_PROJECT_DATA_PATH
if [ ! -d "URBAN100" ] || [ ! -d "SET5" ] || [ ! -d "SET14" ] || [ ! -d "BSDS100" ]; then
    wget http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip
    unzip SR_testing_datasets.zip
    rm SR_testing_datasets.zip
    rm -r historical
    rm -r Manga109
    for dir in "Urban100" "Set5" "Set14" "BSDS100"; do
        dir=${dir%*/} 
        cd $SR_PROJECT_DATA_PATH/$dir
        mkdir HR
        mv *.png HR/
        python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/$dir/HR 2
        python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/$dir/HR 4
        python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/$dir/HR 8
        cd $SR_PROJECT_DATA_PATH
    done
    mv "Urban100" "URBAN100"
    mv "Set5" "SET5"
    mv "Set14" "SET14"
fi

# Build outs directory.
echo $'\nBuilding output directory ...'
cd $SR_PROJECT_HOME
if [ ! -d "outs" ]; then
    mkdir outs
fi

cd $SR_PROJECT_PROJECT_HOME
echo $'\nSuccessfully set up project !'