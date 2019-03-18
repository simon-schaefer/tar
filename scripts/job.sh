#!/bin/bash
#
## otherwise the default shell would be used 
#$ -S /bin/bash
#
## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q short.q@*
# 
## the maximum memory usage of this job
#$ -l h_vmem=20G
# 
## stderr and stdout are merged together to stdout 
#$ -j y
#
# logging directory. preferrably on your scratch 
#$ -o /scratch_net/biwidl215/sischaef/outs/
# 
# calling exectuable. 
source /scratch_net/biwidl215/sischaef/tar/scripts/setup.bash
python3 /scratch_net/biwidl215/sischaef/tar/src/tar/main.py --template IM_AE_TAD_DIV2K --no_augment --verbose --data_range 1-100/1-10 --batch_size 5 --loss HR*10*MSE+LR*1*L1 --print_every 10