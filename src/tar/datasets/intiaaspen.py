#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : NTIAASPEN dataset extension.
# =============================================================================
import os
import glob
import imageio

from tar.dataloader import _IDataset_

class INTIAASPEN(_IDataset_):
    def __init__(self, args, train, scale, name="INTIAASPEN"):
        super(INTIAASPEN, self).__init__(args, name=name, train=train, scale=scale)

    def _scan(self):
        names_hr = glob.glob(self.dir_hr + "/*" + ".png")
        for x in range(0,len(names_hr)):
            names_hr[x] = self.dir_hr + "/hr" + str(x) + ".png"
        # Check if scale == 1, then just return HR images.
        if self.scale == 1: return names_hr, names_hr
        # Otherwise build LR image names for every HR image.
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            filename = filename.replace("hr", "lr")
            names_lr.append(self.dir_lr + "/{}x{}.png".format(filename,self.scale))
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(INTIAASPEN, self)._set_filesystem(dir_data)
        self.directory = os.path.join(dir_data, "NTIAASPEN")
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
        self.dir_lr = os.path.join(self.dir_lr, "X{}".format(self.scale))
