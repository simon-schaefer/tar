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
        self.device = torch.device('cpu' if self.args.cpu else self.args.cuda_device)

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
            if save: self.ckp.clear_results(d)
            v = self.testing_core({}, d, di, save=save, finetuning=finetuning)
            net_applying_times.append(float(v["RUNTIME"]))
            best = self.save_psnr_checkpoint(d, di)
        # Finalizing - Saving and logging.
        self.ckp.write_log("Mean network applying time: {:.5f}ms".format(
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
        if len(self.loader_valid) == 0: return
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        finetuning = epoch >= self.args.fine_tuning
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
            li = di + len(self.loader_test)
            name, scale = d.dataset.name, d.dataset.scale
            if save: self.ckp.clear_results(d)
            self.ckp.write_log("{}x{}".format(name,scale))
            v = {"dataset":"{}".format(name + " "*(10-len(name))),
                 "scale": "x{}".format(scale)}
            v = self.testing_core(v, d, li, save=save, finetuning=finetuning)
            best = self.save_psnr_checkpoint(d, li)
            validations.append(v)
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

    def test_core(self, lr: torch.Tensor, hr: torch.Tensor,
                  finetuning: bool, scale: int) -> dict:
        raise NotImplementedError

    def validation_core(self, v: dict, dataset, di: int,
                        save: bool=False, finetuning: bool=False) -> dict:
        raise NotImplementedError

    def log_description(self) -> List[str]:
        raise NotImplementedError

    def num_epochs(self) -> int:
        raise NotImplementedError

    def scale_current(self, epoch: int) -> int:
        raise NotImplementedError

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

    def prepare(self, data):
        return [a.to(self.device) for a in data[0:2]]

    def step(self):
        if self.args.valid_only:
            self.validation()
            return False
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            if epoch > 1: self.ckp.save(self, epoch)
            return epoch < self.num_epochs() and self.ckp.step()
