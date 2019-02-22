#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Image domain models (autoencoders = AE). 
# =============================================================================
from abc import abstractmethod
import torch
from torch import nn

# =============================================================================
# Abstract autoencoder class - All models should inherit from this class. 
# =============================================================================
class AEAbstract(nn.Module):
 
    def __init__(self):
        super(AEAbstract, self).__init__()
        self.encode = self._encoder()
        self.decode = self._decoder()

    @abstractmethod
    def _encoder(self): 
        pass

    @abstractmethod
    def _decoder(self): 
        pass
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# =============================================================================
# Example autoencoder for mnist dataset (for testing). 
# =============================================================================
class AEMNIST_1D_NOSR(AEAbstract):
    def __init__(self):
        super(AEMNIST_1D_NOSR, self).__init__()

    def _encoder(self): 
        return nn.Sequential(                                       # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),               # b, 16, 10, 10
            nn.ReLU(True),                                          # b, 16, 10, 10
            nn.MaxPool2d(2, stride=2),                              # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),               # b, 8, 3, 3
            nn.ReLU(True),                                          # b, 8, 3, 3
            nn.MaxPool2d(2, stride=1)                               # b, 8, 2, 2
        )

    def _decoder(self): 
        return nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),                 # b, 16, 5, 5
            nn.ReLU(True),                                          # b, 16, 5, 5
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),      # b, 8, 15, 15
            nn.ReLU(True),                                          # b, 8, 15, 15  
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),       # b, 1, 28, 28
            nn.Tanh()                                               # b, 1, 28, 28
        )

class AEMNIST_1D(AEAbstract):
    def __init__(self):
        super(AEMNIST_1D, self).__init__()

    def _encoder(self): 
        return nn.Sequential(                                       # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),               # b, 16, 10, 10
            nn.ReLU(True),                                          # b, 16, 10, 10
            nn.MaxPool2d(2, stride=1),                              # b, 16, 9, 9
            nn.Conv2d(16, 1, 1, stride=2, padding=2),               # b, 1, 7, 7
            nn.ReLU(True),                                          # b, 1, 7, 7
        )

    def _decoder(self): 
        return nn.Sequential(
            nn.ConvTranspose2d(1, 16, 1, stride=2, padding=2),      # b, 16, 9, 9
            nn.ReLU(True),                                          # b, 16, 9, 9
            nn.ConvTranspose2d(16, 1, 2, stride=1),                 # b, 1, 10, 10
            nn.ReLU(True),                                          # b, 1, 10, 10
            nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1),       # b, 1, 28, 28
            nn.Tanh()                                               # b, 1, 28, 28
        )

# =============================================================================
# Model selection function (by string name). 
# =============================================================================
def models_by_name(name: str): 
    return {

        "AEMNIST_1D_NOSR" : AEMNIST_1D_NOSR(), 
        "AEMNIST_1D"      : AEMNIST_1D(), 
    
    }[name]

