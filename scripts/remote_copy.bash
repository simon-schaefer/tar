#!/bin/bash

REMOTE_OUTS="sischaef@biwidl215:/scratch_net/biwidl215/sischaef/outs"
REMOTE_MODS="sischaef@biwidl215:/scratch_net/biwidl215/sischaef/tar/models"
LOCAL="/Users/sele/Projects/tar/outs"

scp -r "$REMOTE_OUTS/*" "$LOCAL/"
scp -r "$REMOTE_MODS/*" "$LOCAL/"
