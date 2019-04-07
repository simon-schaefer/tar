#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : SET14 validation dataset extension. 
# =============================================================================
import os

from tar.dataloader import _Dataset_

class SET14(_Dataset_):
    def __init__(self, args, train, scale, name="SET14"):
        super(SET14, self).__init__(args, name=name, train=train, scale=scale)

    def _scan(self):
        names_hr, names_lr = super(SET14, self)._scan()
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(SET14, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
