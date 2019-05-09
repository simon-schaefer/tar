#!/bin/bash

# Update github repository.
echo $'\nUpdating GitHub repository ...'
cd $SR_PROJECT_PROJECT_HOME/
git stash -a
git fetch
git pull --rebase
git status
