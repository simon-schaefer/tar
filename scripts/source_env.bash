#!/bin/bash

source "scripts/parameters.bash"
source "$DIR_SCRIPTS/header.bash"

if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]
then
    echo "Activating conda environment: $CONDA_ENV_NAME ..."
    source activate $CONDA_ENV_NAME
else
    echo "Conda environment $CONDA_ENV_NAME already active !"
fi
