#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Model training and testing class.
# =============================================================================
import argparse
import os
import math
import numpy as np
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
        self.loader_test  = loader.loader_test
        self.loader_valid = loader.loader_valid
        self.model = model
        self.loss = loss
        self.optimizer = optimization.make_optimizer(model, args)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.error_last = 1e8
        self.test_iter = 1
        self.valid_iter = 1

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
        for batch, (lr, hr, _) in enumerate(self.loader_train[scale]):
            # Load images.
            lr, hr = self.prepare(lr, hr)
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
    # Testing
    # =========================================================================
    def test(self):
        """ Testing function. In parallel (background) evaluate every
        testing dataset stated in the input arguments (args). Log
        results and save model at the end. """
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        finetuning = epoch >= self.args.fine_tuning
        self.model.eval()
        # Iterate over every testing dataset.
        timer_test = misc._Timer_()
        net_applying_times = []
        save = self.args.save_results and self.test_iter%self.args.save_every==0
        self.ckp.write_log(
            "\nTesting {} (saving_results={}) ...".format(self.test_iter,save)
        )
        if save: self.ckp.begin_background()
        # Testing for every dataset, i.e. determine output list of
        # measures such as PSNR, runtime, etc..
        for di, d in enumerate(self.loader_test):
            v = self.testing_core({}, d, di, save=save, finetuning=finetuning)
            net_applying_times.append(float(v["RUNTIME"]))
            best = self.save_psnr_checkpoint(d, di)
        # Finalizing - Saving and logging.
        self.ckp.write_log("Mean network applying time: {:.2f}ms".format(
            np.mean(net_applying_times)*1000
        ))
        if save: self.ckp.end_background()
        self.ckp.write_log(
            "Testing Total: {:.2f}s".format(timer_test.toc()), refresh=True
        )
        self.test_iter += 1
        torch.set_grad_enabled(True)

    # =========================================================================
    # Validation
    # =========================================================================
    def validation(self):
        """ Validation function for validate model after training on several
        (independent) datasets. Determine several metrics as PSNR and runtime
        for different scale factors and visualize them (if saving enabled). """
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        finetuning = epoch >= self.args.fine_tuning
        save = self.args.save_results and self.valid_iter%self.args.save_every==0
        self.model.eval()
        self.ckp.write_log(
            "\nValidation {} (saving_results={}) ...".format(self.valid_iter,save)
        )
        # Validation for every dataset, i.e. determine output list of
        # measures such as PSNR, runtime, etc..
        if save: self.ckp.begin_background()
        validations = []
        for di, d in enumerate(self.loader_valid):
            li = di + len(self.loader_test)
            name, scale = d.dataset.name, d.dataset.scale
            self.ckp.write_log("{}x{}".format(name,scale))
            name = name + " "*(10-len(name))
            v = {"dataset":"{}".format(name),"scale": "x{}".format(scale)}
            v = self.testing_core(v, d, li, save=save, finetuning=finetuning)
            validations.append(v)
            best = self.save_psnr_checkpoint(d, li)
        # Finalizing.
        self.ckp.iter_is_best = best[1][0] + 1 == epoch
        if save: self.ckp.end_background()
        self.ckp.save_validations(validations)
        self.valid_iter += 1
        torch.set_grad_enabled(True)

    # =========================================================================
    # Trainer-Specific Functions
    # =========================================================================
    def optimization_core(self, lr: torch.Tensor, hr: torch.Tensor,
                          finetuning: bool, scale: int) -> optimization._Loss_:
        raise NotImplementedError

    def test_core(self, lr: torch.Tensor, hr: torch.Tensor,
                    finetuning: bool, scale: int) \
                    -> Tuple[List[torch.Tensor], List[str], torch.Tensor, float]:
        raise NotImplementedError

    def validation_core(self, v: dict, dataset, di: int,
                        save: bool=False, finetuning: bool=False) -> dict:
        raise NotImplementedError

    def psnr_description(self) -> List[str]:
        raise NotImplementedError

    def scale_current(self, epoch: int) -> int:
        raise NotImplementedError

    def num_epochs(self) -> int:
        raise NotImplementedError

    # =========================================================================
    # Auxialiary Functions
    # =========================================================================
    def save_psnr_checkpoint(self, d, i: int):
        psnr_descs = self.psnr_description()
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

    def prepare(self, *kwargs):
        device = torch.device('cpu' if self.args.cpu else self.args.cuda_device)

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(a) for a in kwargs]

    def step(self):
        if self.args.valid_only:
            self.validation()
            return False
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            if epoch > 1: self.ckp.save(self, epoch)
            return epoch < self.num_epochs() and self.ckp.step()

# =============================================================================
# TASK-AWARE DOWNSCALING TRAINER.
# =============================================================================
class _Trainer_TAD_(_Trainer_):
    """ Trainer class for training the image super resolution using the task
    aware downscaling method, i.e. downscale to scaled image in autoencoder
    and include difference between encoded features and bicubic image to loss.
    Therefore the trainer assumes the model to have an encoder() and decoder()
    function. """

    def __init__(self, args: argparse.Namespace,
                 loader: dataloader._Data_,
                 model: modules._Model_,
                 loss: optimization._Loss_,
                 ckp: misc._Checkpoint_):

        super(_Trainer_TAD_, self).__init__(args, loader, model, loss, ckp)

    def apply(self, lr, hr, scale, discretize=True, dec_input=None):
        assert misc.is_power2(scale)
        scl, hr_in, hr_out, lr_out = 1, hr.clone(), None, None
        # Downsample image until output (decoded image) has the
        # the right scale, add to LR image and discretize.
        if dec_input is None:
            while scl < scale:
                lr_out = self.model.model.encode(hr_in)
                hr_in, scl = lr_out, scl*2
            if scale in self.args.scales_guidance: lr_out=torch.add(lr_out, lr)
            if discretize:
                lr_out = misc.discretize(
                    lr_out, self.args.rgb_range, not self.args.no_normalize,
                    [self.args.norm_min, self.args.norm_max]
                )
        else: lr_out, scl = dec_input, scale
        # Upscale resulting LR_OUT image until decoding output
        # has the right scale.
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
        rgb_range   = self.args.rgb_range
        nmin, nmax  = self.args.norm_min, self.args.norm_max
        disc_args   = (rgb_range, not self.args.no_normalize, [nmin, nmax])
        pnsrs = np.zeros((num_valid_samples, 3))
        runtimes = np.zeros((num_valid_samples, 1))
        for i, (lr, hr, fname) in enumerate(d):
            lr, hr = self.prepare(lr, hr)
            scale  = d.dataset.scale
            lr_out, hr_out_t = self.apply(lr, hr, scale, discretize=finetuning)
            timer_apply = misc._Timer_()
            _, hr_out_b = self.apply(lr, hr, scale, dec_input=lr)
            runtimes[i] = timer_apply.toc()
            # PSNR - Low resolution image.
            lr_out = misc.discretize(lr_out, *disc_args)
            pnsrs[i,0] = misc.calc_psnr(lr_out, lr, None, nmax-nmin)
            # PSNR - High resolution image (base: lr_out).
            hr_out_t = misc.discretize(hr_out_t, *disc_args)
            pnsrs[i,1] = misc.calc_psnr(hr_out_t, hr, None, nmax-nmin)
            # PSNR - High resolution image (base: lr).
            hr_out_b = misc.discretize(hr_out_b, *disc_args)
            pnsrs[i,2] = misc.calc_psnr(hr_out_b, hr, None, nmax-nmin)
            if save:
                slist = [hr_out_t, hr_out_b, lr_out, lr, hr]
                dlist = ["SHRT", "SHRB", "SLR", "LR", "HR"]
                self.ckp.save_results(slist,dlist,fname[0],d,scale)
            #misc.progress_bar(i+1, num_valid_samples)
        # Logging PSNR values.
        for ip, desc in enumerate(["SLR","SHRT","SHRB"]):
            pnsrs_i = pnsrs[:,ip]
            pnsrs_i.sort()
            v["PSNR_{}_1".format(desc)] = "{:.3f}".format(pnsrs_i[-1])
            v["PSNR_{}_2".format(desc)] = "{:.3f}".format(pnsrs_i[-2])
        log = torch.Tensor([float(v["PSNR_SHRT_1"]),float(v["PSNR_SLR_1"])])
        self.ckp.log[-1, di, :] += log
        v["RUNTIME"] = "{:.5f}".format(np.median(runtimes))
        return v

    def psnr_description(self):
        return ["SHRT", "SLR"]

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
