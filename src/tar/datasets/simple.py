#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : SIMPLE dataset extension. 
# =============================================================================
import os

from tar.dataloader import _Dataset_

class SIMPLE(_Dataset_):
    def __init__(self, args, name='SIMPLE', train=True):
        super(SIMPLE, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_hr, names_lr = super(SIMPLE, self)._scan()
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(SIMPLE, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
