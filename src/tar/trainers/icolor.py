#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task-aware colorization trainer for images.
# =============================================================================
import argparse
import numpy as np
import random

import torch

from tar.trainer import _Trainer_
import tar.miscellaneous as misc

class _Trainer_IColor_(_Trainer_):
    """ Trainer class for training the image coloratization using the task
    aware autoencoder method, i.e. greyish the image in the encoder part and
    colorize it again in decoding while encouting both the difference between
    colorized image and the grey (YCbCr-Y channel) image into account. Therefore
    the trainer assumes the model to have an encoder() and decoder() function."""

    def __init__(self, args, loader, model, loss, ckp):
        super(_Trainer_IColor_, self).__init__(args, loader, model, loss, ckp)

    def apply(self, gry, col, discretize=False, dec_input=None):
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        # Encoding i.e. decolorization.
        if dec_input is None:
            gry_out = self.model.model.encode(col.clone())
            gry_out = torch.add(gry_out, gry)
            if discretize:
                gry_out = misc.discretize(gry_out,[nmin,nmax])
        else: gry_out = dec_input
        # Decoding i.e. colorization
        col_out = self.model.model.decode(gry_out)
        return gry_out, col_out

    def optimization_core(self, gry, col, finetuning, *args):
        gry_out, col_out = self.apply(gry, col, discretize=finetuning)
        loss_kwargs={'COL_GT':col,'COL_OUT':col_out,'GRY_GT':gry,'GRY_OUT':gry_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = len(d)
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        max_samples = self.args.max_test_samples
        psnrs = np.zeros((num_valid_samples, 4))
        runtimes = []
        for i, (gry, col, fname) in enumerate(d):
            gry, col = self.prepare([gry, col])
            scale  = d.dataset.scale
            timer_apply = misc._Timer_()
            gry_out, col_out_t = self.apply(gry, col, discretize=finetuning)
            runtimes.append(timer_apply.toc())
            col_cpy = col.clone()
            col_gry = (col_cpy[:,0,:,:]+col_cpy[:,1,:,:]+col_cpy[:,2,:,:])/3.0
            _,w,h = col_gry.size()
            col_gry = col_gry.view(1,1,w,h)
            _, col_out_g = self.apply(gry, col, dec_input=col_gry)
            timer_apply = misc._Timer_()
            _, col_out_y = self.apply(gry, col, dec_input=gry)
            # PSNR - Grey image.
            gry_out = misc.discretize(gry_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(gry_out, gry, None, nmax-nmin)
            # PSNR - Colored image (base: gry_out).
            col_out_t = misc.discretize(col_out_t, [nmin, nmax])
            psnrs[i,1] = misc.calc_psnr(col_out_t, col, None, nmax-nmin)
            # PSNR - Colored image (base: col_gry).
            col_out_g = misc.discretize(col_out_g, [nmin, nmax])
            psnrs[i,2] = misc.calc_psnr(col_out_g, col, None, nmax-nmin)
            # PSNR - Colored image (base: lr).
            col_out_y = misc.discretize(col_out_y, [nmin, nmax])
            psnrs[i,3] = misc.calc_psnr(col_out_y, col, None, nmax-nmin)
            if save:
                slist = [col_out_t, col_out_y, col_out_g, gry_out, gry, col]
                dlist = ["SCOLT", "SCOLY", "SCOLG", "SGRY", "GRY", "COL"]
                self.ckp.save_results(slist,dlist,fname[0],d,scale)
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        for ip, desc in enumerate(["SGRY","SCOLT","SCOLG","SCOLY"]):
            psnrs_i = psnrs[:,ip]
            psnrs_i.sort()
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mean".format(desc)]="{:.3f}".format(np.mean(psnrs_i))
        log = [float(v["PSNR_{}".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        v["RUNTIME"] = "{:.8f}".format(np.median(runtimes))
        return v

    def log_description(self):
        return ["SCOLT_best", "SCOLT_mean", "SGRY_best", "SGRY_mean"]

    def scale_current(self, epoch):
        return 1

    def num_epochs(self):
        return self.args.epochs_base
