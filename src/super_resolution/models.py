#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of models classes. 
# =============================================================================
from abc import abstractmethod
import argparse

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
        self.idx_scale = 0
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_gpus = torch.cuda.device_count()
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        self.load(ckp.get_path('model'), resume=args.resume, cpu=args.cpu)
        print(self.model, file=ckp.log_file)
        print("... successfully built model module !")

    def forward(self, x: torch.Tensor, idx_scale: int) -> torch.Tensor:
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

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
# Abstract AutoEncoder - All AE-image models should inherit from this class. 
# =============================================================================
class _AEAbstract_(nn.Module):
 
    def __init__(self):
        super(_AEAbstract_, self).__init__()
        self._build_encoder()
        self._build_decoder()

    @abstractmethod
    def _build_encoder(self): 
        pass

    @abstractmethod
    def encode(self, x): 
        pass

    @abstractmethod
    def _build_decoder(self): 
        pass
    
    @abstractmethod
    def decode(self, x): 
        pass

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# =============================================================================
# Example autoencoder for mnist dataset (for testing). 
# =============================================================================
class AEMNIST_1D_NOSR(_AEAbstract_):

    def __init__(self):
        super(AEMNIST_1D_NOSR, self).__init__()

    def _build_encoder(self): 
        self._encode_seq = nn.Sequential(                           # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),               # b, 16, 10, 10
            nn.ReLU(True),                                          # b, 16, 10, 10
            nn.MaxPool2d(2, stride=2),                              # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),               # b, 8, 3, 3
            nn.ReLU(True),                                          # b, 8, 3, 3
            nn.MaxPool2d(2, stride=1)                               # b, 8, 2, 2
        )

    def encode(self, x): 
        return self._encode_seq(x)

    def _build_decoder(self): 
        self._decode_seq = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),                 # b, 16, 5, 5
            nn.ReLU(True),                                          # b, 16, 5, 5
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),      # b, 8, 15, 15
            nn.ReLU(True),                                          # b, 8, 15, 15  
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),       # b, 1, 28, 28
            nn.Tanh()                                               # b, 1, 28, 28
        )
    
    def decode(self, x): 
        return self._decode_seq(x)

class AEMNIST_1D(_AEAbstract_):

    def __init__(self):
        super(AEMNIST_1D, self).__init__()

    def _build_encoder(self): 
        self._encode_seq = nn.Sequential(                           # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),               # b, 16, 10, 10
            nn.ReLU(True),                                          # b, 16, 10, 10
            nn.MaxPool2d(2, stride=1),                              # b, 16, 9, 9
            nn.Conv2d(16, 1, 1, stride=2, padding=2),               # b, 1, 7, 7
            nn.ReLU(True),                                          # b, 1, 7, 7
        )

    def encode(self, x): 
        return self._encode_seq(x)

    def _build_decoder(self): 
        self._decode_seq = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 1, stride=2, padding=2),      # b, 16, 9, 9
            nn.ReLU(True),                                          # b, 16, 9, 9
            nn.ConvTranspose2d(16, 1, 2, stride=1),                 # b, 1, 10, 10
            nn.ReLU(True),                                          # b, 1, 10, 10
            nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1),       # b, 1, 28, 28
            nn.Tanh()                                               # b, 1, 28, 28
        )

    def decode(self, x): 
        return self._decode_seq(x)

# =============================================================================
# Task-aware image downscaling autoencoder model. 
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

class AETAD_1D(_AEAbstract_): 

    def __init__(self):
        super(AETAD_1D, self).__init__() 

    def _build_encoder(self): 
        self._downscaling = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), 
            nn.Conv2d(8, 16, 3, stride=1, padding=1), 
            _ReversePixelShuffle_(downscale_factor=2), 
        )
        self._res_en1 = _Resblock_(64)
        self._res_en2 = _Resblock_(64)
        self._res_en3 = _Resblock_(64)
        self._conv_en1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

    def encode(self, x):                                            # b, 1, 96, 96
        x = self._downscaling(x)                                    # b, 64, 48, 48      
        residual = x 
        x = self._res_en1.forward(x)                                # b, 64, 48, 48 
        x = self._res_en2.forward(x)                                # b, 64, 48, 48 
        x = self._res_en3.forward(x)                                # b, 64, 48, 48 
        x = self._conv_en1.forward(x)                               # b, 64, 48, 48 
        x = torch.add(residual, x)                                  # b, 64, 48, 48   
        x = self._conv_en2.forward(x)                               # b, 1, 48, 48
        return x                                                    

    def _build_decoder(self): 
        self._conv_de1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)  
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64) 
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1), 
            nn.PixelShuffle(upscale_factor=2), 
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )
        
    def decode(self, x): 
        x = self._conv_de1.forward(x)                               # b, 64, 48, 48
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, 48, 48 
        x = self._res_de2.forward(x)                                # b, 64, 48, 48 
        x = self._res_de3.forward(x)                                # b, 64, 48, 48 
        x = self._conv_de2.forward(x)                               # b, 64, 48, 48 
        x = torch.add(residual, x)                                  # b, 64, 48, 48
        x = self._upscaling(x)                                      # b, 1, 96, 96
        return x

# =============================================================================
# Model selection function (by string name). 
# =============================================================================
def models_by_name(name: str): 
    return {

        "AEMNIST_1D_NOSR" : AEMNIST_1D_NOSR(), 
        "AEMNIST_1D"      : AEMNIST_1D(), 
        "AETAD_1D"        : AETAD_1D(),  
    
    }[name]
