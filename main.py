#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : *Main*
# =============================================================================
import tar
import tar.inputs
import torch

tar.miscellaneous.print_header()
args    = tar.inputs.args
ckp     = tar.miscellaneous._Checkpoint_(args)
loader  = tar.dataloader._Data_(args)
loss    = tar.optimization._Loss_(args, ckp) if not args.valid_only else None
model   = tar.modules._Model_(args, ckp)
if args.type == "SCALING":
   trainer = tar.trainers.iscale._Trainer_IScale_(args, loader, model, loss, ckp)
elif args.type == "COLORING":
    trainer = tar.trainers.icolor._Trainer_IColor_(args, loader, model, loss, ckp)
else:
   raise ValueError("Invalid trainer selection {}".format(args.type))

device = torch.device('cpu' if args.cpu else args.cuda_device)
ckp.write_log("Machine: {}".format(torch.cuda.get_device_name(None)))
while trainer.step():
    trainer.train()
    trainer.test()
    trainer.validation()
ckp.done()
