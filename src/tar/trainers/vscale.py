#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task-aware super resolution trainer for videos.
# =============================================================================
import argparse
import numpy as np
import random

import cv2

import torch

from tar.trainer import _Trainer_
import tar.miscellaneous as misc

class _Trainer_VScale_(_Trainer_):
    """ Trainer class for training the video super resolution using the task
    aware downscaling method, i.e. downscale to scaled image in autoencoder
    and include difference between encoded features and bicubic image to loss.
    Therefore the trainer assumes the model to have an encoder() and decoder()
    function. """

    def __init__(self, args, loader, model, loss, ckp):
        super(_Trainer_VScale_, self).__init__(args, loader, model, loss, ckp)

    def apply(self, lr_prev, lr, hr, scale, discretize=False, dec_input=None):
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
        # Apply input images to model and determine output.
        hrm_out = None
        return lr_out, hr_out, hrm_out

    def optimization_core(self, lrs, hrs, finetuning, scale):
        lr1,lr2 = lrs; hr1,hr2 = hrs
        lr_out,hr_out,hrm_out=self.apply(lr1,lr2,hr2,scale,discretize=finetuning)
        # Pass loss variables to optimizer and optimize.
        loss_kwargs = {'HR_GT': hr2,  'HR_OUT': hr_out,
                       'LR_GT': lr2,  'LR_OUT': lr_out,
                       'MODEL_GT': hr2, 'MODEL_OUT': hrm_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = len(d)
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        max_samples = self.args.max_test_samples
        psnrs = np.zeros((num_valid_samples, 3))
        runtimes = []
        for i, data in enumerate(d):
            lrs, hrs = self.prepare(data)
            lr1,lr2 = lrs; hr1,hr2 = hrs
            scale  = d.dataset.scale
            timer_apply = misc._Timer_()
            lr_out,hr_out,hrm_out = self.apply(lr1,lr2,hr2,scale,discretize=finetuning)
            runtimes.append(timer_apply.toc())
            # PSNR - Low resolution image.
            lr_out = misc.discretize(lr_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(lr_out, lr2, None, nmax-nmin)
            # PSNR - High resolution image (base: lr_out).
            hr_out = misc.discretize(hr_out, [nmin, nmax])
            psnrs[i,1] = misc.calc_psnr(hr_out, hr2, None, nmax-nmin)
            # PSNR - Model image.
            psnrs[i,2] = misc.calc_psnr(hrm_out, hr2, None, nmax-nmin)
            if save:
                filename = str(data[0][2][0]).split("_")[0]
                slist = [hr_out, lr_out, hrm_out, lr2, hr2]
                dlist = ["SHR", "SLR", "SHRM", "LR", "HR"]
                self.ckp.save_results(slist,dlist,filename,d,scale)
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        for ip, desc in enumerate(["SLR","SHR","SHRM"]):
            psnrs_i = psnrs[:,ip]
            psnrs_i.sort()
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mdan".format(desc)]="{:.3f}".format(np.median(psnrs_i))
        log = [float(v["PSNR_{}_best".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        v["RUNTIME"] = "{:.8f}".format(np.median(runtimes))
        return v

    def prepare(self, data):
        lr1, hr1 = [a.to(self.device) for a in data[0][0:2]]
        lr2, hr2 = [a.to(self.device) for a in data[1][0:2]]
        return (lr1,lr2), (hr1,hr2)

    def log_description(self):
        return ["SLR", "SHR", "SHRM"]

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
