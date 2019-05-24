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

    def optimization_core(self, lr, hr, finetuning, scale):
        lr_out, hr_out = self.apply(lr, hr, scale, discretize=finetuning)
        loss_kwargs = {'HR_GT': hr, 'HR_OUT': hr_out, 'LR_GT': lr, 'LR_OUT': lr_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = len(d)
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        psnrs = np.zeros((num_valid_samples, 3))
        for i, (lr, hr, fname) in enumerate(d):
            lr, hr = self.prepare([lr, hr])
            scale  = d.dataset.scale
            lr_out, hr_out_t = self.apply(lr, hr, scale, discretize=finetuning)
            _, hr_out_b = self.apply(lr, hr, scale, dec_input=lr)
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
            print(psnrs_i)
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mean".format(desc)]="{:.3f}".format(np.mean(psnrs_i))
        log = [float(v["PSNR_{}".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        # Determine runtimes for up and downscaling and overall.
        runtimes = np.zeros((3, min(len(d),10)))
        for i, (lr, hr, fname) in enumerate(d):
            if i >= runtimes.shape[1]: break
            lr, hr = self.prepare([lr, hr])
            scale  = d.dataset.scale
            timer_apply = misc._Timer_()
            self.apply(lr, hr, scale, discretize=False)
            runtimes[0,i] = timer_apply.toc()
            timer_apply = misc._Timer_()
            self.apply(lr, hr, scale, discretize=False, dec_input=lr)
            runtimes[1,i] = timer_apply.toc()
            runtimes[2,i] = max(runtimes[0,i] - runtimes[1,i], 0.0)
        v["RUNTIME_AL"] = "{:.8f}".format(np.min(runtimes[0,:], axis=0))
        v["RUNTIME_UP"] = "{:.8f}".format(np.min(runtimes[1,:], axis=0))
        v["RUNTIME_DW"] = "{:.8f}".format(np.min(runtimes[2,:], axis=0))
        return v

    def log_description(self):
        return ["SHRT_best", "SHRT_mean", "SHRB_best", "SHRB_mean",
                "SLR_best", "SLR_mean"]
