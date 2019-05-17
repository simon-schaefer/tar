#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : NTIAASPEN dataset extension.
# =============================================================================
import os

from tar.dataloader import _IDataset_

class NTIAASPEN(_VDataset_):
    def __init__(self, args, train, scale, name="BSDS100"):
        super(NTIAASPEN, self).__init__(args, name=name, train=train, scale=scale)

    def _set_filesystem(self, dir_data):
        super(NTIAASPEN, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
        self.dir_lr = os.path.join(self.dir_lr, "X{}".format(self.scale))
