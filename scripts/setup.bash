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

bash $SR_PROJECT_SCRIPTS_PATH/setup/update.bash
bash $SR_PROJECT_SCRIPTS_PATH/setup/build.bash

# Set environment set flag.
export SR_PROJECT_IS_SET="True"
echo "Successfully built environment !"

bash $SR_PROJECT_SCRIPTS_PATH/setup/datasets.bash

cd $SR_PROJECT_PROJECT_HOME
echo $'\nSuccessfully set up project !'
