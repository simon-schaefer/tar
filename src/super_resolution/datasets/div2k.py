#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : DIV2K dataset extension. 
# =============================================================================
from super_resolution.dataloader import _Dataset_

class DIV2K(_Dataset_):
    def __init__(self, args, name='DIV2K', train=True):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if not train and len(data_range) > 1 and not args.test_only: 
            data_range = data_range[1]
        else: 
            data_range = data_range[0]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')