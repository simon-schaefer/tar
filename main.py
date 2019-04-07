#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Main function for superresolution.
# =============================================================================
import tar as _sr_
import torch

_sr_.miscellaneous.print_header()
args    = _sr_.inputs.args
ckp     = _sr_.miscellaneous._Checkpoint_(args)
loader  = _sr_.dataloader._Data_(args)
loss    = _sr_.optimization._Loss_(args, ckp) if not args.valid_only else None
model   = _sr_.modules._Model_(args, ckp)
trainer = None
if args.model_type == "TAD": 
   trainer = _sr_.trainer._Trainer_TAD_(args, loader, model, loss, ckp)
else: 
   raise ValueError("Invalid trainer selection {}".format(args.model_type))
   
device = torch.device('cpu' if args.cpu else args.cuda_device)
ckp.write_log("Machine: {}".format(torch.cuda.get_device_name(None)))
while ckp.ready and not trainer.terminate(): 
    trainer.train()
    trainer.test()
ckp.done()