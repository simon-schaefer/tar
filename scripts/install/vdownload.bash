#!/bin/bash

cd $SR_PROJECT_HOME
HRS="$SR_PROJECT_DATA_PATH/uclOpticalFlow_v1.2/HR"
DOWNSAMPLE_FILE="$SR_PROJECT_PROJECT_HOME/src/datasets/downsample.py"
max_scale=4
## UCL Optical Flow dataset.
echo "UCL Optical Flow dataset ..."
cd $SR_PROJECT_DATA_PATH
if [ ! -d "UCLOPTICALFLOW" ]; then
    wget http://visual.cs.ucl.ac.uk/pubs/flowConfidence/supp/uclOpticalFlow_v1.2.zip
    unzip uclOpticalFlow_v1.2.zip
    rm uclOpticalFlow_v1.2.zip
    cd uclOpticalFlow_v1.2
    if [ ! -d "HR" ]; then
        mv scenes HR
        cd HR
        sample=1
        for f in *; do
            if [ -d "$f" ]; then
                cd $f
                mv 1.png ../"$sample"_1.png
                mv 2.png ../"$sample"_2.png
                mv flow*.png ../"$sample"_flow.png
                cd ..
                rm -r $f
            fi
            sample=$((sample + 1))
        done
        cd ..
    fi
    if [ ! -d "LR_bicubic/X2" ]; then
        for (( s = 2; s <= $max_scale; s=s*2 )); do
            python3 $DOWNSAMPLE_FILE $HRS $s $max_scale "LR_bicubic"
        done
    fi
    cd $SR_PROJECT_DATA_PATH
    mv uclOpticalFlow_v1.2 UCLOPTICALFLOW
fi
cd $SR_PROJECT_HOME
