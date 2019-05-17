#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : BSDS100 validation dataset extension.
# =============================================================================
import os

from tar.dataloader import _IDataset_

class BSDS100(_IDataset_):
    def __init__(self, args, train, scale, name="BSDS100"):
        super(BSDS100, self).__init__(args, name=name, train=train, scale=scale)

    def _set_filesystem(self, dir_data):
        super(BSDS100, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
        self.dir_lr = os.path.join(self.dir_lr, "X{}".format(self.scale))
