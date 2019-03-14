#!/bin/bash

REMOTE="sischaef@biwidl215:/scratch_net/biwidl211/sischaef/outs"
LOCAL="/Users/Sele/Documents/Projects/super_resolution/outs"

scp -r "$REMOTE/*" "$LOCAL/"