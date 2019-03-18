#!/bin/bash

REMOTE="sischaef@biwidl215:/scratch_net/biwidl215/sischaef/outs"
LOCAL="/Users/Sele/Documents/Projects/tar/outs"

scp -r "$REMOTE/*" "$LOCAL/"