#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : DIV2K validation dataset extension. 
# =============================================================================
import os

from tar.dataloader import _Dataset_

class VDIV2K(_Dataset_):
    def __init__(self, args, name='VDIV2K', train=True):
        super(VDIV2K, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_hr, names_lr = super(VDIV2K, self)._scan()
        names_hr = names_hr[self.begin:self.end]
        names_lr = names_lr[self.begin:self.end]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(VDIV2K, self)._set_filesystem(dir_data)
        self.directory = os.path.join(self.args.dir_data, "DIV2K")
        self.dir_hr = os.path.join(self.directory, 'DIV2K_valid_HR')
        self.dir_lr = os.path.join(self.directory, 'DIV2K_valid_LR_bicubic')
