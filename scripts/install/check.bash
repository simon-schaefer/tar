#!/bin/bash

# Download datasets.
echo $'\nCheckings datasets ...'
CHECKING_FILE="$SR_PROJECT_PROJECT_HOME/src/tests/check_dataset.py"

## DIV2K dataset.
echo $'\nDIV2K dataset ...'
DIV2K_HR="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_train_HR"
DIV2K_LR="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_train_LR_bicubic"
python3 $CHECKING_FILE $DIV2K_HR $DIV2K_LR

## DIV2K validation dataset.
echo $'\nDIV2K validation dataset ...'
DIV2Kv_HR="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_valid_HR"
DIV2Kv_LR="$SR_PROJECT_DATA_PATH/DIV2K/DIV2K_valid_LR_bicubic"
python3 $CHECKING_FILE $DIV2Kv_HR $DIV2Kv_LR

# Validation datasets.
echo $'\nValidation datasets ...'
for dir in "URBAN100" "SET5" "SET14" "BSDS100"; do
    echo $"[$dir] ..."
    VALID_HR="$SR_PROJECT_DATA_PATH/$dir/HR"
    VALID_LR="$SR_PROJECT_DATA_PATH/$dir/LR_bicubic"
    python3 $CHECKING_FILE $VALID_HR $VALID_LR
done
