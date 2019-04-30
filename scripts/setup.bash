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

# Parsing input arguments.
usage()
{
    echo "usage: source setup.bash [[[-b] [-d] [-c]] | [-h]]"
}
SHOULD_BUILD=false; SHOULD_UPDATE_DATASET=false; SHOULD_CHECK_DATASET=false; 
while [ "$1" != "" ]; do
    case $1 in
        -b | --build )          SHOULD_BUILD=true
                                shift;;
        -d | --download )       SHOULD_UPDATE_DATASET=true
                                shift;;
        -c | --check )          SHOULD_CHECK_DATASET=true
                                shift;;
        -h | --help )           usage;;
        * )                     usage
    esac
    shift
done

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
bash $SR_PROJECT_SCRIPTS_PATH/install/update.bash

# Build files (install requirements).
if [ "$SHOULD_BUILD" = true ] ; then
    bash $SR_PROJECT_SCRIPTS_PATH/install/build.bash
fi

# Update and check datasets.
if [ "$SHOULD_UPDATE_DATASET" = true ] ; then
    bash $SR_PROJECT_SCRIPTS_PATH/install/download.bash
fi
if [ "$SHOULD_CHECK_DATASET" = true ] ; then
    bash $SR_PROJECT_SCRIPTS_PATH/install/check.bash
fi

cd $SR_PROJECT_PROJECT_HOME
echo $'\nSuccessfully set up project !'
