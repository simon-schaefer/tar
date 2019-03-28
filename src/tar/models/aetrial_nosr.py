#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Example autoencoder for mnist dataset (for testing). 
# =============================================================================
import torch
from torch import nn

def build_net():
    return AETRIAL_1D_NOSR()

class AETRIAL_1D_NOSR(nn.Module):

    def __init__(self):
        super(AETRIAL_1D_NOSR, self).__init__()
        self._encode_seq = nn.Sequential(                           # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),               # b, 16, 10, 10
            nn.ReLU(True),                                          # b, 16, 10, 10
            nn.MaxPool2d(2, stride=2),                              # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),               # b, 8, 3, 3
            nn.ReLU(True),                                          # b, 8, 3, 3
            nn.MaxPool2d(2, stride=1)                               # b, 8, 2, 2
        )
        self._decode_seq = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),                 # b, 16, 5, 5
            nn.ReLU(True),                                          # b, 16, 5, 5
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),      # b, 8, 15, 15
            nn.ReLU(True),                                          # b, 8, 15, 15  
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),       # b, 1, 28, 28
            nn.Tanh()                                               # b, 1, 28, 28
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor: 
        return self._encode_seq(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor: 
        return self._decode_seq(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.decode(self.encode(x))

