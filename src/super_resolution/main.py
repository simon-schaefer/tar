#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Main function for superresolution.
# =============================================================================
import super_resolution as _sr_

args    = _sr_.inputs.args
ckp     = _sr_.miscellaneous._Checkpoint_(args)
loader  = _sr_.dataloader._Data_(args)
loss    = _sr_.optimization._Loss_(args, ckp) if not args.test_only else None
model   = _sr_.modules._Model_(args, ckp)
trainer = None
if args.model.find("TAD") >= 0: 
    trainer = _sr_.trainer._Trainer_TAD_(args, loader, model, loss, ckp)
else: 
    trainer = _sr_.trainer._Trainer_(args, loader, model, loss, ckp)

while ckp.ready and not trainer.terminate(): 
    trainer.train()
    trainer.test()
ckp.done()