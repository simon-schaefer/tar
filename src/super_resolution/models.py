#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Model superclasses and custom building modules. 
# =============================================================================
from abc import abstractmethod
import argparse
from importlib import import_module

import torch
from torch import nn
import torch.utils.model_zoo

import super_resolution.miscellaneous as misc

class _Model_(nn.Module):
    ''' Model front end module including parallization (adapting to available
    hardware) as well as saving/loading functions. 
    All models should inherit from this (abstract) model class. '''

    def __init__(self, args: argparse.Namespace, ckp: misc._Checkpoint_):
        super(_Model_, self).__init__()
        print("Building model module ...")
        # Set parameters from input arguments. 
        self.scale = args.scale
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_gpus = torch.cuda.device_count()
        self.save_models = args.save_models
        module = import_module('models.' + args.model.lower())
        self.model = module.build_net().to(self.device)
        self.load(ckp.get_path('model'), resume=args.resume, cpu=args.cpu)
        print(self.model, file=ckp.log_file)
        print("... successfully built model module !")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training: 
            if self.n_gpus > 1:
                return torch.nn.parallel.data_parallel(
                    self.model, x, range(self.n_GPUs)
                )
            else:
                return self.model(x)
        else:
            return self.model.forward(x)

    # =========================================================================
    # Saving and Loading 
    # =========================================================================
    def save(self, directory: str, epoch: int, is_best: bool=False):
        ''' Save model as latest version, as epoch version and (if is_best flag
        is set to True) as best version. '''
        save_dirs = [directory + "/model_latest.pt"]
        if is_best:
            save_dirs.append(directory + "/model_best.pt")
        if self.save_models:
            save_dirs.append(directory + "/model_{}.pt".format(epoch))
        # Save model under given paths. 
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, directory: str, resume: int=-1, cpu: bool=False):
        ''' Load model from directory, either the latest version (resume = -1)
        or from a specific epoch (resume = epoch) to device. '''
        load_from = None
        kwargs = {'map_location': lambda storage, loc: storage} if cpu else {}
        if resume == -1:
            load_from = torch.load(directory + "/model_latest.pt", **kwargs)
        else:
            load_from = torch.load(directory + "/model_{}.pt".format(resume), **kwargs)
        self.model.load_state_dict(load_from, strict=False)

# =============================================================================
# Customly implemented building blocks (nn.Modules). 
# =============================================================================
class _Resblock_(nn.Module): 
    ''' Residual convolutional block consisting of two convolutional 
    layers, a RELU activation in between and a residual connection from 
    start to end. The inputs size (=s) is therefore contained. The number 
    of channels is contained as well, but can be adapted (=c). '''

    __constants__ = ['channels']

    def __init__(self, c):
        super(_Resblock_, self).__init__()
        self.filter_block = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True),     # b, c, s, s
            nn.ReLU(True),                                          # b, c, s, s
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True)      # b, c, s, s
        )
        self.channels = c

    def forward(self, x): 
        return x + self.filter_block(x)

    def extra_repr(self):
        return 'channels={}'.format(self.channels)

class _ReversePixelShuffle_(nn.Module): 
    ''' Reverse pixel shuffeling module, i.e. rearranges elements in a tensor 
    of shape (*, C, H*r, W*r) to (*, C*r^2, H, W). '''

    __constants__ = ['downscale_factor']

    def __init__(self, downscale_factor):
        super(_ReversePixelShuffle_, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return self.inv_pixel_shuffle(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)

    @staticmethod
    def inv_pixel_shuffle(input, downscale_factor):
        batch_size, in_channels, height, width = input.size()
        out_channels = in_channels * (downscale_factor ** 2)
        height //= downscale_factor
        width //= downscale_factor
        # Reshape input to new shape. 
        input_view = input.contiguous().view(
            batch_size, in_channels, downscale_factor, downscale_factor,
            height, width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous() 
        return shuffle_out.view(batch_size, out_channels, height, width)

# =============================================================================
# AutoEncoder model class - Adding encoding and decoding. 
# =============================================================================
class _AEModel_(_Model_):
 
    def __init__(self, args: argparse.Namespace, ckp: misc._Checkpoint_):
        super(_AEAbstract_, self).__init__(args, ckp)

    def encode(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model.encode(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model.decode(x)

