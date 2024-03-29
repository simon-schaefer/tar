#!/bin/bash
#
## otherwise the default shell would be used
#$ -S /bin/bash
#
## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q gpu.24h.q@*
#
## the maximum memory usage of this job
#$ -l gpu=1 -l h_vmem=40G
#
## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferrably on your scratch
#$ -o /scratch_net/biwidl215/sischaef/outs/
#
# calling exectuable.
source /scratch_net/biwidl215/sischaef/tar/scripts/setup.bash --build
cd /scratch_net/biwidl215/sischaef/tar/src/cic/
python3 main.py
