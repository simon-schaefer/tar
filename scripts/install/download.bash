#!/bin/bash

# Build outs directory.
echo $'\nBuilding output directory ...'
cd $SR_PROJECT_HOME
if [ ! -d "outs" ]; then
    mkdir outs
fi

# Download datasets.
echo $'\nDownloading datasets ...'
cd $SR_PROJECT_HOME
if [ ! -d "data" ]; then
    mkdir data
fi
cd data
DOWNSAMPLE_FILE="$SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py"
max_scale=16
## DIV2K dataset.
echo "DIV2K dataset ..."
if [ ! -d "DIV2K" ]; then
    mkdir DIV2K
fi
cd DIV2K
HRS="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_train_HR"
if [ ! -d "DIV2K_train_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    unzip DIV2K_train_HR.zip
    rm DIV2K_train_HR.zip
fi
# if [ ! -d "DIV2K_train_LR_bicubic/X2" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
#     unzip DIV2K_train_LR_bicubic_X2.zip
#     rm DIV2K_train_LR_bicubic_X2.zip
# fi
# if [ ! -d "DIV2K_train_LR_bicubic/X4" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
#     unzip DIV2K_train_LR_bicubic_X4.zip
#     rm DIV2K_train_LR_bicubic_X4.zip
# fi
# if [ ! -d "DIV2K_train_LR_bicubic/X8" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
#     unzip DIV2K_train_LR_x8.zip
#     rm DIV2K_train_LR_x8.zip
#     mv DIV2K_train_LR_x8 DIV2K_train_LR_bicubic/X8
#     python3 $CHECKING_FILE $HRS "DIV2K_train_LR_bicubic"
# fi
if [ ! -d "DIV2K_train_LR_bicubic/X16" ]; then
    for (( s = 2; s <= $max_scale; s=s*2 )); do
        python3 $DOWNSAMPLE_FILE $HRS $s $max_scale "DIV2K_train_LR_bicubic"
    done
fi

# SIMPLE dataset.
# echo "SIMPLE dataset ..."
# cd $SR_PROJECT_DATA_PATH
# if [ ! -d "SIMPLE" ]; then
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/create_simple_dataset.py
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/SIMPLE/HR 2
#     python3 $SR_PROJECT_PROJECT_HOME/src/tests/downsample_dataset.py $SR_PROJECT_DATA_PATH/SIMPLE/HR 4
# fi

## DIV2K validation dataset.
echo "DIV2K validation dataset ..."
HRS="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_valid_HR"
if [ ! -d "DIV2K_valid_HR" ]; then
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
fi
# if [ ! -d "DIV2K_valid_LR_bicubic/X2" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
#     unzip DIV2K_valid_LR_bicubic_X2.zip
#     rm DIV2K_valid_LR_bicubic_X2.zip
# fi
# if [ ! -d "DIV2K_valid_LR_bicubic/X4" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
#     unzip DIV2K_valid_LR_bicubic_X4.zip
#     rm DIV2K_valid_LR_bicubic_X4.zip
# fi
# if [ ! -d "DIV2K_valid_LR_bicubic/X8" ]; then
#     wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip
#     unzip DIV2K_valid_LR_x8.zip
#     rm DIV2K_valid_LR_x8.zip
#     mv DIV2K_valid_LR_x8 DIV2K_valid_LR_bicubic/X8
# fi
if [ ! -d "DIV2K_valid_LR_bicubic/X16" ]; then
    for (( s = 2; s <= $max_scale; s=s*2 )); do
        python3 $DOWNSAMPLE_FILE $HRS $s $max_scale "DIV2K_valid_LR_bicubic"
    done
fi

# Validation datasets.
echo "Validation datasets ..."
cd $SR_PROJECT_DATA_PATH
if [ ! -d "URBAN100" ] || [ ! -d "SET5" ] || [ ! -d "SET14" ] || [ ! -d "BSDS100" ]
then
    wget http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip
    unzip SR_testing_datasets.zip
    rm SR_testing_datasets.zip; rm -r historical; rm -r Manga109
    for dir in "Urban100" "Set5" "Set14" "BSDS100"; do
        dir=${dir%*/}
        cd $SR_PROJECT_DATA_PATH/$dir
        mkdir HR
        mv *.png HR/
        for (( s = 2; s <= $max_scale; s=s*2 )); do
            python3 $DOWNSAMPLE_FILE $SR_PROJECT_DATA_PATH/$dir/HR $s $max_scale
        done

        cd $SR_PROJECT_DATA_PATH
    done
    mv "Urban100" "URBAN100"; mv "Set5" "SET5"; mv "Set14" "SET14"
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
