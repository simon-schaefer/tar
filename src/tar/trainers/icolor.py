#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task aware colorization trainer for images.
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
        self.ckp.write_log("... successfully built icolor trainer !")

    def apply(self, gry, col, scale=None, discretize=False, mode="all"):
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        assert mode in ["all", "up", "down"]
        # Encoding i.e. decolorization and decoding i.e. colorization
        def _decolorization(col):
            gry_out = self.model.model.encode(col.clone())
            gry_out = torch.add(gry_out, gry)
            if discretize: gry_out = misc.discretize(gry_out,[nmin,nmax])
            return gry_out
        def _colorization(gry):
            return self.model.model.decode(gry.clone())
        # In case of down- or upscaling only perform only part scalings.
        if mode == "down": return _decolorization(col)
        elif mode == "up": return _colorization(gry)
        # Otherwise perform down- and upscaling and return both tensors.
        gry_out = _decolorization(col)
        col_out = _colorization(gry_out)
        return gry_out, col_out

    def optimization_core(self, gry, col, finetuning, *args):
        if not self.args.no_task_aware:
            gry_out, col_out = self.apply(gry, col,discretize=finetuning)
        else:
            gry_out, col_out = gry, self.apply(gry, col, mode="up")
        loss_kwargs={'COL_GT':col,'COL_OUT':col_out,'GRY_GT':gry,'GRY_OUT':gry_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = d.dataset.sample_size
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        psnrs = np.zeros((num_valid_samples, 2))
        for i, (gry, col, fname) in enumerate(d):
            gry, col = self.prepare([gry, col])
            scale  = d.dataset.scale
            gry_out, col_out_t = self.apply(gry, col, discretize=finetuning)
            # PSNR - Grey image.
            gry_out = misc.discretize(gry_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(gry_out, gry, None, nmax-nmin)
            # PSNR - Colored image (base: gry_out).
            col_out_t = misc.discretize(col_out_t, [nmin, nmax])
            psnrs[i,1] = misc.calc_psnr(col_out_t, col, None, nmax-nmin)
            if save:
                slist = [col_out_t, gry_out, gry, col]
                dlist = ["SCOLT", "SGRY", "GRY", "COL"]
                self.ckp.save_results(slist,dlist,fname[0],d,scale)
            if save and i % self.args.n_threads == 0:
                self.ckp.end_background()
                self.ckp.begin_background()
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        v = self.logging_core(psnrs=psnrs, di=di, v=v)
        # Determine runtimes for up and downscaling and overall.
        v = self.runtime_core(d, v)
        return v

    def psnr_description(self):
        return ["SGRY","SCOLT"]

    def log_description(self):
        return ["SCOLT_best", "SCOLT_mean", "SGRY_best", "SGRY_mean"]

    def scale_current(self, epoch):
        return 1

    def num_epochs(self):
        return self.args.epochs_base
