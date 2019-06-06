#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task aware super resolution trainer for videos.
# =============================================================================
import argparse
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

    def apply(self,lrs,hrs,scale,discretize=False,dec_input=[None,None,None],mode="all"):
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        assert mode in ["all", "up", "down"]
        if mode == "up": assert not all([x is None for x in dec_input])
        # Downscaling.
        lr0,lr1,lr2 = lrs; hr0,hr1,hr2 = hrs
        if mode == "down":
            return super(_Trainer_VExternal_, self).apply(
                lr1, hr1, scale, discretize, dec_input[1], mode="down"
            )
        # Apply input images to model and determine output.
        lr0_out, lr1_out, lr2_out = dec_input
        if lr0_out is None:
            lr0_out = super(_Trainer_VExternal_, self).apply(
            lr0, hr0, scale, discretize, mode="down")
        if lr1_out is None:
            lr1_out = super(_Trainer_VExternal_, self).apply(
            lr1, hr1, scale, discretize, mode="down")
        if lr2_out is None:
            lr2_out = super(_Trainer_VExternal_, self).apply(
            lr2, hr2, scale, discretize, mode="down")
        lr0e = misc.discretize(lr0_out.clone(),[nmin,nmax])
        lr1e = misc.discretize(lr1_out.clone(),[nmin,nmax])
        lr2e = misc.discretize(lr2_out.clone(),[nmin,nmax])
        hre_out = self._external.apply(lr0e.clone(),lr1e.clone(),lr2e.clone())
        if mode == "up": return hre_out
        return lr1_out, hre_out

    def optimization_core(self, lrs, hrs, finetuning, scale):
        if not self.args.no_task_aware:
            lr_out, hrm_out = self.apply(lrs,hrs,scale,discretize=finetuning)
        else:
            lr_out  = lrs[1].clone()
            hrm_out = self.apply(lrs, hrs, scale, dec_input=lrs, mode="up")
        # Pass loss variables to optimizer and optimize.
        loss_kwargs = { 'LR_GT': lr1,  'LR_OUT': lr_out,
                       'EXT_GT': hr1, 'EXT_OUT': hrm_out}
        loss = self.loss(loss_kwargs)
        return loss

    def testing_core(self, v, d, di, save=False, finetuning=False):
        num_valid_samples = d.dataset.sample_size
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        psnrs = np.zeros((num_valid_samples, 3))
        for i, (lrs, hrs, fnames) in enumerate(d):
            fnames   = [str(x)[2:-3] for x in fnames]
            lrs, hrs = self.prepare([lrs, hrs])
            lr0,lr1,lr2 = lrs; hr0,hr1,hr2 = hrs
            scale  = d.dataset.scale
            if not self.args.no_task_aware:
                lr_out,hrm_out = self.apply(lrs,hrs,scale,discretize=finetuning)
            else:
                lr_out  = lrs[1].clone()
                hrm_out = self.apply(lrs, hrs, scale, dec_input=lrs, mode="up")
            # PSNR - Low resolution image.
            lr_out = misc.discretize(lr_out, [nmin, nmax])
            psnrs[i,0] = misc.calc_psnr(lr_out, lr1, None, nmax-nmin)
            # PSNR - Model image.
            psnrs[i,1] = misc.calc_psnr(hrm_out, hr1, None, nmax-nmin)
            if save:
                slist = [lr_out, hrm_out, lr1, hr1]
                dlist = ["SLR", "SHRET", "LR", "HR"]
                self.ckp.save_results(slist,dlist,fnames[1],d,scale)
            if save and i % self.args.n_threads == 0:
                self.ckp.end_background()
                self.ckp.begin_background()
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        v = self.logging_core(psnrs=psnrs, di=di, v=v)
        return v

    def perturbation_core(self, d, eps):
        num_testing_samples = min(len(d), 10)
        psnrs_t = np.zeros((num_testing_samples,len(eps)))
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        for id, (lrs, hrs, fname) in enumerate(d):
            if id >= num_testing_samples: break
            lrs, hrs = self.prepare([lrs, hrs])
            lr0,lr1,lr2 = lrs; hr0,hr1,hr2 = hrs
            scale  = d.dataset.scale
            lr_out = self.apply(lrs,hrs,scale,discretize=True,mode="down")
            for ie, e in enumerate(eps):
                error = torch.normal(mean=0.0,std=torch.ones(lr_out.size())*e)
                lr_out = lr_out.clone() + error.to(self.device)
                hr_out_eps = self.apply(lrs, hrs, scale,
                                        dec_input=[None,lr_out,None], mode="up")
                hr_out_eps = misc.discretize(hr_out_eps, [nmin, nmax])
                psnrs_t[id,ie] = misc.calc_psnr(hr_out_eps, hr1, None, nmax-nmin)
        return psnrs_t.mean(axis=0)

    def prepare(self, data):
        lrs = [a.to(self.device) for a in data[0]]
        hrs = [a.to(self.device) for a in data[1]]
        return tuple(lrs), tuple(hrs)

    def psnr_description(self):
        return ["SLR","SHRET"]

    def log_description(self):
        return ["SHRET_best", "SHRET_mean", "SLR_best", "SLR_mean"]

    def scale_current(self, epoch):
        if not self.args.valid_only:
            assert self.args.scales_train == self.args.scales_valid
        return self.args.scales_train[0]

    def num_epochs(self):
        return self.args.epochs_base
