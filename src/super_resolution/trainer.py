#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Model training and testing class. 
# =============================================================================
import argparse
import os
import math

import torch

import super_resolution.dataloader as dataloader
import super_resolution.miscellaneous as misc
import super_resolution.modules as modules
import super_resolution.optimization as optimization

class _Trainer_(object):
    ''' Training front end including training and testing functions according 
    to the input arguments. Also includes automated logging. '''

    def __init__(self, args: argparse.Namespace, 
                 loader: dataloader._Data_, 
                 model: modules._Model_, 
                 loss: optimization._Loss_, 
                 ckp: misc._Checkpoint_):
    
        super(_Trainer_, self).__init__()
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.optimizer = optimization.make_optimizer(model, args)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.error_last = 1e8

    # =========================================================================
    # Training
    # =========================================================================
    def train(self):
        ''' Training function for one epoch. Automated logging, using
        optimizer and loss stated in the __init__ (their state is loaded
        and updated automatically). '''
        self.optimizer.schedule()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log('[Epoch {}]\tLearning rate: {}'.format(epoch, lr))
        self.loss.start_log()
        self.model.train()
        # Iterate over all batches in epoch. 
        timer_data, timer_model = misc._Timer_(), misc._Timer_()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            # Load images. 
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            # Optimization core. 
            self.optimizer.zero_grad()
            loss = self.optimization_core(lr, hr)
            loss.backward()
            if self.args.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            # Logging (if printable epoch).
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        # Finalizing - Save error and logging. 
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        print("... finalizing epoch {}.".format(epoch))

    # =========================================================================
    # Testing
    # =========================================================================
    def test(self):
        ''' Testing function. In parallel (background) evaluate every 
        testing dataset stated in the input arguments (args). Log 
        results and save model at the end. '''
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test))
        )
        self.model.eval()
        # Iterate over every testing dataset. 
        timer_test = misc._Timer_()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename in d:
                lr, hr = self.prepare(lr, hr)
                hr_out = self.model(hr)
                hr_out = misc.discretize(
                    hr_out, self.args.rgb_range, not self.args.no_normalize
                )
                save_list = [hr_out]
                self.ckp.log[-1, idx_data] += misc.calc_psnr(
                    hr_out, hr, self.scale, self.args.rgb_range
                )
                if self.args.save_gt:
                    save_list.extend([lr, hr])
                if self.args.save_results:
                    self.ckp.save_results(save_list, filename[0], d, self.scale)
            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.scale,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )
        # Finalizing - Saving and logging. 
        self.ckp.write_log("Forward: {:.2f}s".format(timer_test.toc()))
        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.write_log("Saving states ...")
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def optimization_core(self, lr: torch.Tensor, hr: torch.Tensor) -> optimization._Loss_: 
        hr_out = self.model(hr)
        loss_kwargs = {'HR_GT': hr, 'HR_OUT': hr_out}
        loss = self.loss(loss_kwargs)
        return loss

    def prepare(self, *kwargs):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        return [x.to(device) for x in kwargs]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

# =============================================================================
# TASK-AWARE DOWNSCALING TRAINER. 
# =============================================================================
class _Trainer_TAD_(_Trainer_): 
    ''' Trainer class for training the image super resolution using the task
    aware downscaling method, i.e. downscale to scaled image in autoencoder
    and include difference between encoded features and bicubic image to loss.
    Therefore the trainer assumes the model to have an encoder() and decoder()
    function. '''

    def __init__(self, args: argparse.Namespace, 
                 loader: dataloader._Data_, 
                 model: modules._Model_, 
                 loss: optimization._Loss_, 
                 ckp: misc._Checkpoint_):

        super(_Trainer_TAD_, self).__init__(args, loader, model, loss, ckp)
    
    def optimization_core(self, lr: torch.Tensor, hr: torch.Tensor) -> optimization._Loss_: 
        lr_out = self.model.model.encode(hr)
        hr_out = self.model.model.decode(lr_out)
        loss_kwargs = {'HR_GT': hr, 'HR_OUT': hr_out, 'LR_GT': lr, 'LR_OUT': lr_out}
        loss = self.loss(loss_kwargs)
        return loss  


