#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : BSDS100 validation dataset extension. 
# =============================================================================
import os

from tar.dataloader import _Dataset_

class BSDS100(_Dataset_):
    def __init__(self, args, name='BSDS100', train=True):
        super(BSDS100, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_hr, names_lr = super(BSDS100, self)._scan()
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(BSDS100, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
