#!/bin/bash

HOST=sischaef@biwidl215
REMOTE_OUTS=/scratch_net/biwidl215/sischaef/outs
REMOTE_MODS=/scratch_net/biwidl215/sischaef/tar/models
LOCAL=/Users/sele/Projects/tar/outs

for path in $REMOTE_OUTS $REMOTE_MODS; do
    dirs=$(ssh $HOST ls $path)
    for dir in $dirs; do
        mkdir $LOCAL/$dir
        for ext in ".pdf" ".txt" ".pt" ".pth" ".csv"; do
            scp $HOST:$path/$dir/*$ext $LOCAL/$dir/
        done
        scp -r $HOST:$path/$dir/model $LOCAL/$dir/
    done
done
