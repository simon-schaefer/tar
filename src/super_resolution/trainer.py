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
import super_resolution.models as models
import super_resolution.optimization as optimization

class _Trainer_(object):
    ''' Training front end including training and testing functions according 
    to the input arguments. Also includes automated logging. '''

    def __init__(self, args: argparse.Namespace, 
                 loader: dataloader._Data_, 
                 model: models._Model_, 
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
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        # Iterate over all batches in epoch. 
        timer_data, timer_model = misc.timer(), misc.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            # Load images. 
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            # Optimization core. 
            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
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
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        # Iterate over every testing dataset. 
        timer_test = misc.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename, _ in tqdm(d, ncols=80):
                lr, hr = self.prepare(lr, hr)
                sr = self.model(lr)
                sr = misc.discretize(sr, self.args.rgb_range)
                save_list = [sr]
                self.ckp.log[-1, idx_data, 0] += misc.calc_psnr(
                    sr, hr, self.scale, self.args.rgb_range, dataset=d
                )
                if self.args.save_gt:
                    save_list.extend([lr, hr])
                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list, scale)
            self.ckp.log[-1, idx_data, 0] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.scale,
                    self.ckp.log[-1, idx_data, 0],
                    best[0][idx_data, 0],
                    best[1][idx_data, 0] + 1
                )
            )
        # Finalizing - Saving and logging. 
        self.ckp.write_log("Forward: {:.2f}s".format(timer_test.toc()))
        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.write_log("Saving states ...")
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        return [tensor.to(device)(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

