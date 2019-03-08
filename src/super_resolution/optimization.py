#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Optimization and Loss implementations. 
# =============================================================================
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn

import super_resolution.miscellaneous as misc

# =============================================================================
# OPTIMIZER. 
# =============================================================================
def make_optimizer(model: torch.nn.Module, args: argparse.Namespace): 
    ''' Building the optimizer for model training purposes, given the 
    dictionary of input arguments (argparse). Also add functionalities like 
    saving/loading a optimizer state. '''
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'ADAM': 
        optimizer_class = torch.optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    else: 
        raise ValueError("Invalid optimizer class = %s !" % str(args.optimizer))
    # Build scheduler (learning rate adaption). 
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {"milestones": milestones, "gamma": args.gamma}
    scheduler_class = torch.optim.lr_scheduler.MultiStepLR

    # Building optimizer front end class based on chosen optimizer class. 
    class _Optimizer_(optimizer_class): 
        ''' Optimizer front end class appending the chosen optimizer class 
        by utility functions as saving/loading the internal state and also 
        adding an automatic scheduler (adaptive learning rate). '''

        def __init__(self, *args, **kwargs):
            super(_Optimizer_, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def schedule(self):
            self.scheduler.step()

        def load(self, directory, epoch=1):
            self.load_state_dict(torch.load(directory + "/optimizer.pth"))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def save(self, directory):
            torch.save(self.state_dict(), directory + "/optimizer.pth")

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    # Return made optimization class after initializing the scheduler. 
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = _Optimizer_(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

# =============================================================================
# LOSS. 
# =============================================================================
class _Loss_(nn.modules.loss._Loss):
    ''' Building loss front end module that is creating a loss function based 
    on the argument list (argparse) and adapting it to the available devices. '''

    def __init__(self, args: argparse.Namespace, ckp: misc._Checkpoint_):
        super(_Loss_, self).__init__()
        # Reading loss function from arguments. Every loss is given seperated
        # by '+' in the 'loss' argument while one loss is defined by the method
        # (e.g. L1, MSE) and its weight (i.e. lambda). Additionally, the argument
        # includes information about the to be compared elements, i.e. whether 
        # they are high or low resolution images (HR, LR).  
        ckp.write_log("Building loss module ...")
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'): 
            input_type, weight, loss_type = loss.split('*')
            loss_function = None
            if loss_type == "L1": 
                loss_function = nn.L1Loss()
            elif loss_type == "MSE": 
                loss_function = nn.MSELoss()
            else: 
                raise ValueError("Invalid loss type %s !" % loss_type)
            # Append to overall loss array. 
            self.loss.append({
                "weight"    : float(weight), 
                "desc"      : "{}-{}".format(input_type, loss_type), 
                "function"  : loss_function})
            self.loss_module.append(loss_function)
            ckp.write_log("... adding %s loss with weight=%f and input type=%s" \
                    % (loss_type, float(weight), input_type))
        # For logging purposes add additional element to loss 
        # containing the sum of all "sub-losses". 
        self.loss.append({"desc": "TOTAL", "weight": 0.0, "function": None})
        # Load loss function to given device. 
        self.n_gpus = args.n_gpus
        device = torch.device("cpu" if args.cpu else args.cuda_device)
        self.loss_module.to(device)
        if args.precision == "half": 
            self.loss_module.half()
        if not args.cpu and self.n_gpus > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(self.n_GPUs)
            )
        # Build logging and load previous log (if required).  
        self.log = torch.Tensor()
        if args.load != "": self.load(ckp.dir, cpu=args.cpu)
        ckp.write_log("... successfully built loss module !")

    def forward(self, kwargs):
        ''' Given the required input arguments determine every single
        loss as well as the total loss, return and add to logging. 
        Expected inputs: {"LR_GT": low resolution ground truth, 
                          "LR_OUT": low resolution output
                          "HR_GT": high resolution ground truth, 
                          "HR_OUT": high resolution }
        '''
        # Search for required inputs for specific loss type.
        for l  in self.loss: 
            input_type = l["desc"].split("-")[0]
            if input_type == "HR" and \
            (not "HR_GT" in kwargs.keys() or not "HR_OUT" in kwargs.keys()): 
                raise ValueError("Loss missing HR arguments !")
            if input_type == "LR" and \
            (not "LR_GT" in kwargs.keys() or not "LR_OUT" in kwargs.keys()): 
                raise ValueError("Loss missing LR arguments !")
        # Determine loss given loss function. 
        losses = []
        for i, l in enumerate(self.loss):
            input_type = l["desc"].split("-")[0]
            if l["function"] is not None:
                x, y = kwargs[input_type+"_OUT"], kwargs[input_type+"_GT"]
                loss = l["function"](x, y)
                effective_loss = l["weight"] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        self.log[-1, -1] += loss_sum.item()
        return loss_sum

    # =========================================================================
    # Saving and Loading 
    # =========================================================================
    def save(self, directory: str):
        ''' Save internal state and logging in given directory. '''
        torch.save(self.state_dict(), directory + "/loss.pt")
        torch.save(self.log, directory + "/loss.pt")

    def load(self, directory: str, cpu: bool=False):
        ''' Load internal state and logging from given directory and redo  
        logging state (simulate previous steps). '''
        kwargs = {"map_location": lambda storage, loc: storage} if cpu else {}
        self.load_state_dict(torch.load(directory + "/loss.pt"), **kwargs)
        self.log = torch.load(directory + "/loss_log.pt")

    # =========================================================================
    # Logging, plotting, displaying 
    # =========================================================================
    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches: int):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch: int) -> str:
        ''' Build loss description string containing a list of all losses
        and their according normalized tensor. '''
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['desc'], c / n_samples))
        return "".join(log)

    def get_total_loss(self) -> float: 
        return self.log[-1, -1]

    def plot_loss(self, directory: str, epoch: int, threshold: float=1e3):
        ''' Plot loss curves of every internal loss function and store
        the resulting figure in given directory. To avoid badly scaled 
        loss plots the values are thresholded, i.e. every loss above the 
        threshold value is set to the threshold value. '''
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = l["desc"]
            fig = plt.figure()
            plt.title(label)
            losses = self.log[:, i].numpy()
            losses[losses > threshold] = threshold
            plt.plot(axis, losses, label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(directory + "/loss_{}.pdf".format(l['desc']))
            plt.close(fig)

    def get_loss_module(self) -> nn.ModuleList:
        ''' Return loss modules (depending on number of gpus they are either 
        stored as single instance or in parallel). '''
        if self.n_gpus == 1:
            return self.loss_module
        else:
            return self.loss_module.module