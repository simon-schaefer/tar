#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task-aware super resolution trainer for images.
# =============================================================================
import argparse
import numpy as np
import random

import torch

from tar.trainer import _Trainer_
import tar.miscellaneous as misc

class _Trainer_IScale_(_Trainer_):
    """ Trainer class for training the image super resolution using the task
    aware downscaling method, i.e. downscale to scaled image in autoencoder
    and include difference between encoded features and bicubic image to loss.
    Therefore the trainer assumes the model to have an encoder() and decoder()
    function. """

    def __init__(self, args, loader, model, loss, ckp):
        super(_Trainer_IScale_, self).__init__(args, loader, model, loss, ckp)
        self.ckp.write_log("... successfully built iscale trainer !")

    def apply(self, lr, hr, scale, discretize=False, dec_input=None):
        assert misc.is_power2(scale)
        scl, hr_in, hr_out, lr_out = 1, hr.clone(), None, None
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        # Downsample image until output (decoded image) has the
        # the right scale, add to LR image and discretize.
        if dec_input is None:
            while scl < scale:
                lr_out = self.model.model.encode(hr_in)
                hr_in, scl = lr_out, scl*2
            if scale in self.args.scales_guidance: lr_out=torch.add(lr_out, lr)
            if discretize:
                lr_out = misc.discretize(lr_out,[nmin,nmax])
        else: lr_out, scl = dec_input, scale
        # Upscale resulting LR_OUT image until decoding output has right scale.
        hr_out = lr_out.clone()
        while scl > 1:
            hr_out = self.model.model.decode(hr_out)
            scl    = scl//2
        return lr_out, hr_out

    def optimization_core(self, lr, hr, finetuning, scale):
        lr_out, hr_out = self.apply(lr, hr, scale, discretize=finetuning)
        loss_kwargs = {'HR_GT': hr, 'HR_OUT': hr_out, 'LR_GT': lr, 'LR_OUT': lr_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = len(d)
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        psnrs = np.zeros((num_valid_samples, 3))
        runtimes = []
        for i, (lr, hr, fname) in enumerate(d):
            lr, hr = self.prepare([lr, hr])
            scale  = d.dataset.scale
            lr_out, hr_out_t = self.apply(lr, hr, scale, discretize=finetuning)
            timer_apply = misc._Timer_()
            _, hr_out_b = self.apply(lr, hr, scale, dec_input=lr)
            runtimes.append(timer_apply.toc())
            # PSNR - Low resolution image.
            lr_out = misc.discretize(lr_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(lr_out, lr, None, nmax-nmin)
            # PSNR - High resolution image (base: lr_out).
            hr_out_t = misc.discretize(hr_out_t, [nmin, nmax])
            psnrs[i,1] = misc.calc_psnr(hr_out_t, hr, None, nmax-nmin)
            # PSNR - High resolution image (base: lr).
            hr_out_b = misc.discretize(hr_out_b, [nmin, nmax])
            psnrs[i,2] = misc.calc_psnr(hr_out_b, hr, None, nmax-nmin)
            if save:
                slist = [hr_out_t, hr_out_b, lr_out, lr, hr]
                dlist = ["SHRT", "SHRB", "SLR", "LR", "HR"]
                self.ckp.save_results(slist,dlist,fname[0],d,scale)
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        for ip, desc in enumerate(["SLR","SHRT","SHRB"]):
            psnrs_i = psnrs[:,ip]
            psnrs_i.sort()
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mean".format(desc)]="{:.3f}".format(np.mean(psnrs_i))
        log = [float(v["PSNR_{}".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        v["RUNTIME"] = "{:.8f}".format(np.median(runtimes))
        return v

    def log_description(self):
        return ["SHRT_best", "SHRT_mean", "SHRB_best", "SHRB_mean",
                "SLR_best", "SLR_mean"]

    def scale_current(self, epoch):
        scalestrain  = self.args.scales_train
        ebase, ezoom = self.args.epochs_base, self.args.epochs_zoom
        if epoch < ebase: return scalestrain[0]
        else: return scalestrain[(epoch-ebase)//ezoom+1]

    def num_epochs(self):
        ebase, ezoom = self.args.epochs_base, self.args.epochs_zoom
        if len(self.args.scales_train) == 1: return ebase
        n_zooms = len(self.args.scales_train) - 1
        return ebase + n_zooms*ezoom
