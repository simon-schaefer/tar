#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task-aware super resolution trainer for videos.
# =============================================================================
import argparse
import importlib
import numpy as np
import random

import torch

from tar.trainer import _Trainer_
import tar.miscellaneous as misc

class _Trainer_VExternal_(_Trainer_):
    """ Trainer class for training the video super resolution using the task
    aware downscaling method, i.e. downscale to scaled image in autoencoder
    and include difference between encoded features and bicubic image to loss.
    Therefore the trainer assumes the model to have an encoder() and decoder()
    function. """

    def __init__(self, args, loader, model, loss, ckp):
        super(_Trainer_VExternal_, self).__init__(args,loader,model,loss,ckp)
        external = self.args.external
        if external == "": raise ValueError("External module must not be empty !")
        use_gpu  = not self.args.cpu
        self._external = self.load_module(external,self.scale_current(0),use_gpu)
        self.ckp.write_log("... successfully built vscale trainer !")

    def apply(self,lr_prev,lr,lr_next,hr,scale,discretize=False,dec_input=None):
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
        lr_ext = misc.discretize(lr_out.clone(),[nmin,nmax])
        hre_out = self._external.apply(lr_prev, lr_ext, lr_next)
        return lr_out, hr_out, hre_out

    def optimization_core(self, lrs, hrs, finetuning, scale):
        lr0,lr1,lr2 = lrs; hr0,hr1,hr2 = hrs
        lr_out,hr_out,hrm_out = self.apply(lr0,lr1,lr2,hr1,
                                           scale,discretize=finetuning)
        # Pass loss variables to optimizer and optimize.
        loss_kwargs = {'HR_GT': hr2,  'HR_OUT': hr_out,
                       'LR_GT': lr2,  'LR_OUT': lr_out,
                       'EXT_GT': hr2, 'EXT_OUT': hrm_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = len(d)
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        psnrs = np.zeros((num_valid_samples, 4))
        runtimes = []
        for i, data in enumerate(d):
            lrs, hrs = self.prepare(data)
            lr0,lr1,lr2 = lrs; hr0,hr1,hr2 = hrs
            scale  = d.dataset.scale
            timer_apply = misc._Timer_()
            lr_out,hr_out,hrm_out = self.apply(lr0,lr1,lr2,hr1,
                                               scale,discretize=finetuning)
            runtimes.append(timer_apply.toc())
            _,_,hrm_out2 = self.apply(lr0,lr1,lr2,hr1,
                                     scale,discretize=finetuning,
                                     dec_input=lr1)
            # PSNR - Low resolution image.
            lr_out = misc.discretize(lr_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(lr_out, lr2, None, nmax-nmin)
            # PSNR - High resolution image (base: lr_out).
            hr_out = misc.discretize(hr_out, [nmin, nmax])
            psnrs[i,1] = misc.calc_psnr(hr_out, hr2, None, nmax-nmin)
            # PSNR - Model image.
            psnrs[i,2] = misc.calc_psnr(hrm_out, hr2, None, nmax-nmin)
            psnrs[i,3] = misc.calc_psnr(hrm_out2, hr2, None, nmax-nmin)
            if save:
                filename = str(data[0][2][0]).split("_")[0]
                slist = [hr_out, lr_out, hrm_out, hrm_out2, lr2, hr2]
                dlist = ["SHR", "SLR", "SHRET", "SHREB", "LR", "HR"]
                self.ckp.save_results(slist,dlist,filename,d,scale)
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        for ip, desc in enumerate(["SLR","SHR","SHRET","SHREB"]):
            psnrs_i = psnrs[:,ip]
            psnrs_i.sort()
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mean".format(desc)]="{:.3f}".format(np.mean(psnrs_i))
        log = [float(v["PSNR_{}".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        v["RUNTIME"] = "{:.8f}".format(np.median(runtimes))
        return v

    def prepare(self, data):
        lr0, hr0 = [a.to(self.device) for a in data[0][0:2]]
        lr1, hr1 = [a.to(self.device) for a in data[1][0:2]]
        lr2, hr2 = [a.to(self.device) for a in data[2][0:2]]
        return (lr0,lr1,lr2), (hr0,hr1,hr2)

    def log_description(self):
        return ["SLR_best", "SLR_mean", "SHR_best", "SHR_mean",
                "SHRET_best", "SHRET_mean", "SHREB_best", "SHREB_mean"]

    def scale_current(self, epoch):
        assert self.args.scales_train == self.args.scales_valid
        return self.args.scales_train[0]

    def num_epochs(self):
        return self.args.epochs_base

    @staticmethod
    def load_module(model_name, scale, use_gpu=True):
        m = importlib.import_module(model_name.lower() + ".apply")
        return getattr(m, model_name)(scale, use_gpu)
