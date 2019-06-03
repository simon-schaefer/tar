#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Model training and validation class.
# =============================================================================
import argparse
import os
import importlib
import math
import numpy as np
import random
from typing import List, Tuple

import torch

import tar.dataloader as dataloader
import tar.miscellaneous as misc
import tar.modules as modules
import tar.optimization as optimization

class _Trainer_(object):
    """ Training front end including training and testing functions according
    to the input arguments. Also includes automated logging. """

    def __init__(self, args: argparse.Namespace,
                 loader: dataloader._Data_,
                 model: modules._Model_,
                 loss: optimization._Loss_,
                 ckp: misc._Checkpoint_):

        super(_Trainer_, self).__init__()
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_valid = loader.loader_valid
        self.check_datasets()
        self.model = model
        self.loss = loss
        self.optimizer = optimization.make_optimizer(model, args)
        self.error_last = 1e8
        self.valid_iter = 1
        self.device = torch.device('cpu' if self.args.cpu else self.args.cuda_device)
        self.ckp.write_log("Building trainer module ...")

    # =========================================================================
    # Training
    # =========================================================================
    def train(self):
        """ Training function for one epoch. Automated logging, using
        optimizer and loss stated in the __init__ (their state is loaded
        and updated automatically). """
        self.optimizer.schedule()
        epoch = self.optimizer.get_last_epoch() + 1
        finetuning = epoch >= self.args.fine_tuning
        scale = self.scale_current(epoch)
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            "\n[Epoch {}]\tLearning rate: {}\tFinetuning: {}\t Scale: x{}".format(
            epoch, lr, finetuning, scale
        ))
        self.loss.start_log()
        self.model.train()
        # Iterate over all batches in epoch.
        timer_data, timer_model = misc._Timer_(), misc._Timer_()
        for batch, data in enumerate(self.loader_train[scale]):
            # Load images.
            lr, hr = self.prepare(data)
            timer_data.hold()
            timer_model.tic()
            # Optimization core.
            self.optimizer.zero_grad()
            loss = self.optimization_core(lr, hr, finetuning, scale)
            loss.backward()
            if self.args.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            # Logging (if printable epoch).
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log("[{}/{}]\t{}\t{:.1f}+{:.1f}s".format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train[scale].dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        # Finalizing - Save error and logging.
        self.loss.end_log(len(self.loader_train[scale]))
        self.error_last = self.loss.log[-1, -1]
        print("... epoch {} with train loss {}".format(epoch, self.error_last))

    # =========================================================================
    # Testing and Validation
    # =========================================================================
    def validation(self):
        """ Validation function for validate model after training on several
        (independent) datasets. Determine several metrics as PSNR and runtime
        for different scale factors and visualize them (if saving enabled). """
        if len(self.loader_valid) == 0: return
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        finetuning = epoch >= self.args.fine_tuning or self.args.valid_only
        save = self.args.save_results and self.valid_iter%self.args.save_every==0
        save = save or self.args.valid_only
        self.model.eval()
        self.ckp.write_log(
            "\nValidation {} (saving_results={}) ...".format(self.valid_iter,save)
        )
        # Validation for every dataset, i.e. determine output list of
        # measures such as PSNR, runtime, etc..
        if save: self.ckp.begin_background()
        timer_valid = misc._Timer_()
        validations = []
        for di, d in enumerate(self.loader_valid):
            name, scale = d.dataset.name, d.dataset.scale
            if save: self.ckp.clear_results(d)
            self.ckp.write_log("{}x{}".format(name,scale))
            v = {"dataset":"{}".format(name + " "*(10-len(name))),
                 "scale": "x{}".format(scale)}
            v = self.testing_core(v, d, di, save=save, finetuning=finetuning)
            best = self.save_psnr_checkpoint(d, di)
            validations.append(v)
        # Determine average runtime.
        if save:
            self.ckp.write_log(
                "Validation {} (runtime test) ...".format(self.valid_iter)
            )
            runtime_al = np.mean([float(v["RUNTIME_AL"]) for v in validations])
            runtime_up = np.mean([float(v["RUNTIME_UP"]) for v in validations])
            runtime_dw = np.mean([float(v["RUNTIME_DW"]) for v in validations])
            self.ckp.save_runtimes(runtime_al,runtime_up,runtime_dw)
        # Perturbation/Noise testing i.e. perturb random SLR image in dataset
        # in different degrees and measure drop of PSNR.
        if save:
            self.ckp.write_log(
                "Validation {} (perturbation test) ...".format(self.valid_iter)
            )
            eps     = np.linspace(0.0,self.args.max_eps,num=10).tolist()
            psnrs_t = np.zeros((len(self.loader_valid),len(eps)))
            labels  = []
            for di, d in enumerate(self.loader_valid):
                name, scale = d.dataset.name, d.dataset.scale
                psnrs_t[di,:] = self.perturbation_core(d, eps)
                labels.append("{}x{}".format(name,scale))
            self.ckp.save_pertubation(eps, psnrs_t, labels)
        # Finalizing.
        self.ckp.iter_is_best = best[1][0] + 1 == epoch
        if save: self.ckp.end_background()
        self.ckp.write_log(
            "Validation Total: {:.2f}s".format(timer_valid.toc()), refresh=True
        )
        self.ckp.save_validations(validations)
        self.valid_iter += 1
        torch.set_grad_enabled(True)

    # =========================================================================
    # Trainer-Specific Functions
    # =========================================================================
    def optimization_core(self, lr: torch.Tensor, hr: torch.Tensor,
                          finetuning: bool, scale: int) -> optimization._Loss_:
        raise NotImplementedError

    def testing_core(self, v: dict, dataset, di: int,
                     save: bool=False, finetuning: bool=False) -> dict:
        raise NotImplementedError

    def apply(self, lr, hr, scale, discretize=False, dec_input=None, mode="all"):
        assert misc.is_power2(scale)
        assert mode in ["all", "up", "down"]
        if mode == "up": assert not dec_input is None
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        iter_scale  = self.args.iter_scale_factor
        assert scale % iter_scale == 0
        # Downsample image until output (decoded image) has the
        # the right scale, add to LR image and discretize.
        def _downsample(hr, scale):
            hr_in = hr.clone()
            scl   = 1
            while scl < scale:
                lr_out = self.model.model.encode(hr_in)
                hr_in, scl = lr_out, scl*iter_scale
            if scale in self.args.scales_guidance: lr_out=torch.add(lr_out, lr.clone())
            if discretize: lr_out = misc.discretize(lr_out,[nmin,nmax])
            return lr_out
        def _upsample(lr, scl):
            hr_out = lr.clone()
            while scl > 1:
                hr_out = self.model.model.decode(hr_out)
                scl    = scl//iter_scale
            return hr_out
        # In case of down- or upscaling only perform only part scalings.
        if mode == "down":
            if dec_input is not None: return lr.clone()
            else:                     return _downsample(hr, scale)
        elif mode == "up":            return _upsample(dec_input, scale)
        # Otherwise perform down- and upscaling and return both tensors.
        lr_out = _downsample(hr, scale)
        hr_out = _upsample(lr_out, scale)
        return lr_out, hr_out

    def logging_core(self, psnrs, di:int, v: dict) -> dict:
        for ip, desc in enumerate(self.psnr_description()):
            psnrs_i = psnrs[:,ip]
            psnrs_i.sort()
            v["PSNR_{}_best".format(desc)]="{:.3f}".format(psnrs_i[-1])
            v["PSNR_{}_mean".format(desc)]="{:.3f}".format(np.mean(psnrs_i))
        log = [float(v["PSNR_{}".format(x)]) for x in self.log_description()]
        self.ckp.log[-1, di, :] += torch.Tensor(log)
        return v

    def perturbation_core(self, d, eps: List[float]):
        num_testing_samples = min(len(d), 10)
        psnrs_t = np.zeros((num_testing_samples,len(eps)))
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        for id, (lr, hr, fname) in enumerate(d):
            if id >= num_testing_samples: break
            lr, hr = self.prepare([lr, hr])
            scale  = d.dataset.scale
            lr_out = self.apply(lr, hr, scale, discretize=True, mode="down")
            for ie, e in enumerate(eps):
                error = torch.normal(mean=0.0,std=torch.ones(lr_out.size())*e)
                lr_out = lr_out.clone() + error.to(self.device)
                hr_out_eps = self.apply(lr,hr,scale,dec_input=lr_out,mode="up")
                hr_out_eps = misc.discretize(hr_out_eps, [nmin, nmax])
                psnrs_t[id,ie] = misc.calc_psnr(hr_out_eps, hr, None, nmax-nmin)
        return psnrs_t.mean(axis=0)

    def runtime_core(self, d, v: dict) -> dict:
        runtimes = np.zeros((2, min(len(d),10)))
        for i, (lr, hr, fname) in enumerate(d):
            if i >= runtimes.shape[1]: break
            lr, hr = self.prepare([lr, hr])
            scale  = d.dataset.scale
            timer_all = misc._Timer_()
            self.apply(lr, hr, scale, discretize=False, mode="all")
            runtimes[0,i] = timer_all.toc()
            timer_up = misc._Timer_()
            self.apply(lr, hr, scale, discretize=False, dec_input=lr, mode="up")
            runtimes[1,i] = timer_up.toc()
        v["RUNTIME_AL"] = "{:.8f}".format(np.median(runtimes[0,:]))
        v["RUNTIME_UP"] = "{:.8f}".format(np.median(runtimes[1,:]))
        v["RUNTIME_DW"] = str(max(float(v["RUNTIME_AL"])-float(v["RUNTIME_UP"]), 0.0))
        return v

    def psnr_description(self) -> List[str]:
        raise NotImplementedError

    def log_description(self) -> List[str]:
        raise NotImplementedError

    def num_epochs(self) -> int:
        ebase, ezoom = self.args.epochs_base, self.args.epochs_zoom
        if len(self.args.scales_train) == 1: return ebase
        n_zooms = len(self.args.scales_train) - 1
        return ebase + n_zooms*ezoom

    def scale_current(self, epoch: int) -> int:
        scalestrain  = self.args.scales_train
        ebase, ezoom = self.args.epochs_base, self.args.epochs_zoom
        if epoch < ebase: return scalestrain[0]
        else: return scalestrain[(epoch-ebase)//ezoom+1]

    # =========================================================================
    # Auxialiary Functions
    # =========================================================================
    def save_psnr_checkpoint(self, d, i: int):
        psnr_descs = self.log_description()
        assert len(psnr_descs) == self.ckp.log.shape[-1]
        for ip, desc in enumerate(psnr_descs):
            best = self.ckp.log[:,:,ip].max(0)
            self.ckp.write_log(
                "{}\t[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})".format(
                    desc, d.dataset.name, d.dataset.scale,
                    self.ckp.log[-1, i, ip], best[0][i], best[1][i] + 1
                )
            )
        return best

    @staticmethod
    def load_module(model_name, scale, use_gpu=True):
        assert len(model_name.split("-")) == 2
        program, model = model_name.split("-")
        m = importlib.import_module(program.lower() + ".apply")
        return getattr(m, program.upper())(model, scale, use_gpu)

    def check_datasets(self):
        def _check(d, format):
            if not d.format == format:
                raise ValueError("Dataset {} has wrong format, is {} \
                but should be {} !".format(d.name, d.format, format))
        for d in self.loader_valid: _check(d.dataset, self.args.format)
        if self.args.valid_only: return True
        for d in self.loader_train.values(): _check(d.dataset, self.args.format)
        return True

    def prepare(self, data):
        return [a.to(self.device) for a in data[0:2]]

    def step(self):
        num_descs = len(self.log_description())
        if self.args.valid_only:
            self.ckp.step(nlogs=num_descs)
            self.validation()
            return False
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            if epoch > 1: self.ckp.save(self, epoch)
            return epoch < self.num_epochs() and self.ckp.step(nlogs=num_descs)

# # =========================================================================
# # Testing
# # =========================================================================
# def test(self):
#     """ Testing function. In parallel (background) evaluate every
#     testing dataset stated in the input arguments (args). Log
#     results and save model at the end. """
#     torch.set_grad_enabled(False)
#     epoch = self.optimizer.get_last_epoch() + 1
#     finetuning = epoch >= self.args.fine_tuning
#     self.model.eval()
#     # Iterate over every testing dataset.
#     timer_test = misc._Timer_()
#     net_applying_times = []
#     save = self.args.save_results and self.test_iter%self.args.save_every==0
#     self.ckp.write_log(
#         "\nTesting {} (saving_results={}) ...".format(self.test_iter,save)
#     )
#     if save: self.ckp.begin_background()
#     # Testing for every dataset, i.e. determine output list of
#     # measures such as PSNR, runtime, etc..
#     for di, d in enumerate(self.loader_test):
#         if save: self.ckp.clear_results(d)
#         v = self.testing_core({}, d, di, save=save, finetuning=finetuning)
#         net_applying_times.append(float(v["RUNTIME"]))
#         best = self.save_psnr_checkpoint(d, di)
#     # Finalizing - Saving and logging.
#     self.ckp.write_log("Mean network applying time: {:.5f}ms".format(
#         np.mean(net_applying_times)*1000
#     ))
#     if save: self.ckp.end_background()
#     self.ckp.write_log(
#         "Testing Total: {:.2f}s".format(timer_test.toc()), refresh=True
#     )
#     self.test_iter += 1
#     torch.set_grad_enabled(True)
