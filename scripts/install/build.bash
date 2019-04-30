#!/bin/bash

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

echo "Successfully built environment !"
