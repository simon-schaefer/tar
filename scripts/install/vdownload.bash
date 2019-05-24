#!/bin/bash

cd $SR_PROJECT_DATA_PATH
DOWNSAMPLE_FILE="$SR_PROJECT_PROJECT_HOME/src/datasets/downsample.py"
## ntia_Aspen dataset.
echo "Ntia_Aspen video dataset ..."
if [ ! -d "NTIAASPEN" ]; then
    git clone https://github.com/LongguangWang/SOF-VSR
    HRS="$SR_PROJECT_DATA_PATH/NTIAASPEN/HR"
    LRS="$SR_PROJECT_DATA_PATH/NTIAASPEN/LR_bicubic"
    mv SOF-VSR/data/train/ntia_Aspen NTIAASPEN
    cd NTIAASPEN
    mv hr HR
    mkdir LR_bicubic
    mv lr_x4_BI LR_bicubic/X4
    for filename in $LRS/X4/*.png; do
        mv $filename ${filename/.png/x4.png}
    done
    python3 $DOWNSAMPLE_FILE $HRS 2 2 "LR_bicubic"
    for filename in $LRS/X2/*.png; do
        mv $filename ${filename/hr/lr}
    done
    cd $SR_PROJECT_DATA_PATH
    rm -rf SOF-VSR
fi

## calendar dataset.
echo "Vid4 video dataset ..."
if [ ! -d "CALENDAR" ] || [ ! -d "CITY" ] || [ ! -d "FOLIAGE" ] || [ ! -d "WALK" ]
then
    if [ ! -d "FRVSR_VID4" ]; then
        wget https://owncloud.tuebingen.mpg.de/index.php/s/2AFqCHjHFtqezR9/download
        mv download FRVSR_VID4.zip
        unzip FRVSR_VID4.zip
        rm FRVSR_VID4.zip
    fi
    for dir in "CALENDAR" "CITY" "FOLIAGE" "WALK"; do
        if [ ! -d "$dir" ]; then
            mkdir -p $dir/HR
        fi
    done
    cp FRVSR_VID4/HR/calendar/*.png $SR_PROJECT_DATA_PATH/CALENDAR/HR/.
    cp FRVSR_VID4/HR/city/*.png $SR_PROJECT_DATA_PATH/CITY/HR/.
    cp FRVSR_VID4/HR/foliage/*.png $SR_PROJECT_DATA_PATH/FOLIAGE/HR/.
    cp FRVSR_VID4/HR/walk/*.png $SR_PROJECT_DATA_PATH/WALK/HR/.
    for dir in "CALENDAR" "CITY" "FOLIAGE" "WALK"; do
        for (( s = 2; s <= 4; s=s*2 )); do
            python3 $DOWNSAMPLE_FILE $SR_PROJECT_DATA_PATH/$dir/HR $s 4
        done
    done
fi

# cd $SR_PROJECT_HOME
# HRS="$SR_PROJECT_DATA_PATH/uclOpticalFlow_v1.2/HR"
# DOWNSAMPLE_FILE="$SR_PROJECT_PROJECT_HOME/src/datasets/downsample.py"
# max_scale=4
# ## UCL Optical Flow dataset.
# echo "UCL Optical Flow dataset ..."
# cd $SR_PROJECT_DATA_PATH
# if [ ! -d "UCLOPTICALFLOW" ]; then
#     wget http://visual.cs.ucl.ac.uk/pubs/flowConfidence/supp/uclOpticalFlow_v1.2.zip
#     unzip uclOpticalFlow_v1.2.zip
#     rm uclOpticalFlow_v1.2.zip
#     cd uclOpticalFlow_v1.2
#     if [ ! -d "HR" ]; then
#         mv scenes HR
#         cd HR
#         sample=1
#         for f in *; do
#             if [ -d "$f" ]; then
#                 cd $f
#                 mv 1.png ../"$sample"_1.png
#                 mv 2.png ../"$sample"_2.png
#                 mv flow*.png ../"$sample"_flow.png
#                 cd ..
#                 rm -r $f
#             fi
#             sample=$((sample + 1))
#         done
#         cd ..
#     fi
#     if [ ! -d "LR_bicubic/X2" ]; then
#         for (( s = 2; s <= $max_scale; s=s*2 )); do
#             python3 $DOWNSAMPLE_FILE $HRS $s $max_scale "LR_bicubic"
#         done
#     fi
#     cd $SR_PROJECT_DATA_PATH
#     mv uclOpticalFlow_v1.2 UCLOPTICALFLOW
# fi
# cd $SR_PROJECT_HOME
