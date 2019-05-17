#!/bin/bash

HOST=sischaef@biwidl215
REMOTE_OUTS=/scratch_net/biwidl215/sischaef/outs
LOCAL=/Users/sele/Projects/tar/outs

scp -r $HOST:$REMOTE_OUTS/* $LOCAL
