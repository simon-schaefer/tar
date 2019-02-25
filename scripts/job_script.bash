#!/bin/bash

#CUDA_HOME=/scratch_net/bmicdl03/libs/cuda-8.0-bmic
PROJECT_HOME=$1
VIRTUAL_ENV_PATH=$2

## otherwise the default shell would be used
#$ -S /bin/bash
 
## Pass environment variables of workstation to GPU node 
#$ -V
 
### <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue
### -q short.q@*
 
## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=8G
 
## stderr and stdout are merged together to stdout
#$ -j y

## logging directory. preferrably on your scratch
#$ -o /scratch_net/biwidl211/sischaef/outs

## send mail on job's end and abort
#$ -m bea

# python virtual environment
export PATH="/home/sischaef/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source $VIRTUAL_ENV_PATH/bin/activate
 
# logging
/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

## EXECUTION OF PYTHON CODE:
python3 $PROJECT_HOME/$3 $4

# logging
/bin/echo finished at: `date`