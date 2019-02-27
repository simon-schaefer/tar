#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Loading data front end class. 
# =============================================================================
import argparse
import imageio
import numpy as np
import os
import glob
import random
import skimage.color as sc
import typing

import torch

from super_resolution.external import ms_data_loader as ms_loader

class _Data_(object):
    ''' Data loading class which allocates all given training and testing
    dataset stated in the input arguments to a loader (and concatenates 
    them for training). The resulting loader_test and loader_train can be 
    used to load batches from the datasets. '''

    def __init__(self, args: argparse.Namespace):
        # Load testing dataset. In order to get seperated testing results, 
        # from each dataset (due to comparability reasons) the testing 
        # datasets are each loaded individually. 
        self.loader_test = []
        for dataset in args.data_test:
            datasets.append(self.load_dataset(args, dataset, train=False))
            self.loader_test.append(ms_loader.MSDataLoader(
                args, testset,
                batch_size=1, shuffle=False, pin_memory=not args.cpu
            ))
        if not args.test_only:
            return
        # Load training dataset, if not testing only. For training several
        # datasets are trained in one process and therefore, each given 
        # training dataset is concatinated to one large dataset. 
        self.loader_train = None
        datasets = []
        for dataset in args.data_train:
            datasets.append(self.load_dataset(args, dataset, train=True))
        self.loader_train = ms_loader.MSDataLoader(
            args, ms_loader.MSConcatDataset(datasets),
            batch_size=args.batch_size, shuffle=True, pin_memory=not args.cpu
        )

    @staticmethod 
    def load_dataset(args: argparse.Namespace, name: str, train: bool) -> _Dataset_: 
        ''' Load dataset from module (in datasets directory). Every module loaded
        should inherit from the _Dataset_ class defined below. '''
        m = import_module('datasets.' + name.lower())
        datasets.append(getattr(m, name)(args), train=train)

# =============================================================================
# DATASET EXTENSION. 
# =============================================================================
class _Dataset_(torch.utils.data.Dataset):
    ''' Extension class for torch dataset module, in order to find, search, 
    load, preprocess and batch data from datasets. '''

    def __init__(self, args: argparse.Namespace, name: str="", train: bool=True):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = args.scale
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.directory, 'bin')
            os.makedirs(path_bin, exist_ok=True)
        else: 
            raise ValueError("Invalid file extension %s !" % str(args.ext))    
        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0:
            self.images_hr, self.images_lr = list_hr, list_lr
        else: 
            raise ValueError("Invalid file extension %s !" % str(args.ext))
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # =========================================================================
    # Handling the filesystem 
    # =========================================================================
    def _scan(self) -> typing.Tuple[typing.List[str], typing.List[str]]:
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        return names_hr, names_lr
    
    def _set_filesystem(self, directory: str):
        self.directory = os.path.join(directory, self.name)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext: str, img: str, f):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def _load_file(self, idx: int) -> typing.Tuple[np.ndarray, np.ndarray, str]:
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img':
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        else: 
            raise ValueError("Invalid file extension %s !" % str(self.args.ext))
        return lr, hr, filename

    # =========================================================================
    # Getter for images
    # =========================================================================
    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor, str]:
        # Load image file. 
        lr, hr, filename = self._load_file(idx)
        # Cut patches from file. 
        def _get_patch(self, lr, hr):
            scale = self.scale
            if self.train:
                ih, iw = lr.shape[:2]
                p = scale if multi else 1
                tp = p * self.args.patch_size
                ip = tp // scale
                ix = random.randrange(0, iw - ip + 1)
                iy = random.randrange(0, ih - ip + 1)
                tx, ty = scale * ix, scale * iy
                lr = lr[iy:iy + ip, ix:ix + ip, :]
                hr = hr[ty:ty + tp, tx:tx + tp, :]
            else:
                ih, iw = lr.shape[:2]
                hr = hr[0:ih * scale, 0:iw * scale]
            return lr, hr

        pair = _get_patch(lr, hr)
        # Augment patches (if flag is set). 
        if not self.args.no_augment: 
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5

            def _augment(img):
                if hflip: img = img[:, ::-1, :]
                if vflip: img = img[::-1, :, :]
                if rot90: img = img.transpose(1, 0, 2)
                
                return img

            pair = [_augment(x) for x in pair]
        # Set right number of channels. 
        def _set_channel(img):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            c = img.shape[2]
            if n_channels == 1 and c == 3:
                img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
            elif n_channels == 3 and c == 1:
                img = np.concatenate([img] * n_channels, 2)
            return img

        pair_t = [_set_channel(x) for x in pair_t]
        # Convert to torch tensor and return. 
        def _np2Tensor(img, rgb_range=255):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)
            return tensor

        pair_t = [_np2Tensor(x, rgb_range=self.args.rgb_range) for x in pair_t]
        return pair_t[0], pair_t[1], filename

    # =========================================================================
    # Miscellaneous
    # =========================================================================
    def __len__(self) -> int:
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx: int) -> int:
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx