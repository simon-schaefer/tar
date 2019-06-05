#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task aware image downscaling autoencoder model - SCALING.
# =============================================================================
import torch
from torch import nn

from tar.modules import _Resblock_, _ReversePixelShuffle_

def build_net():
    return AETAD_SKIP()

class AETAD_SKIP(nn.Module):

    def __init__(self):
        super(AETAD_SKIP, self).__init__()
        # Build encoding part.
        self._downscaling = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en1 = _Resblock_(64)
        self._res_en2 = _Resblock_(64)
        self._res_en3 = _Resblock_(64)
        self._conv_en1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        x = self._downscaling(x)                                    # b, 64, p/2, p/2
        residual = x
        x = self._res_en1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en2.forward(x)                                # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        residual2 = x
        x = self._res_en3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_en1(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual2, x)                                 # b, 64, p/2, p/2
        x = self._conv_en2(x)                                       # b, 3, p/2, p/2
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        residual2 = x
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual2, x)                                  # b, 64, p/2, p/2
        x = self._upscaling(x)                                      # b, 3, p, p
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
