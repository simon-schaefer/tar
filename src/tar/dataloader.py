#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Loading data front end class. 
# =============================================================================
import argparse
import imageio
import importlib
import numpy as np
import os
import glob
import random
import skimage.color as sc
from typing import List, Tuple

from torch import from_numpy, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import tar.miscellaneous as misc

# =============================================================================
# DATASET EXTENSION. 
# =============================================================================
class _Dataset_(Dataset):
    """ Extension class for torch dataset module, in order to find, search, 
    load, preprocess and batch data from datasets. """

    def __init__(self, args, train: bool, scale: int, name: str=""):
        # Initialize super dataset class. 
        super(_Dataset_, self).__init__()
        # Set input parameters. 
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = scale
        # Determining training/testing data range. 
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if not train and len(data_range) > 1 and not args.valid_only: 
            data_range = data_range[1]
        else: 
            data_range = data_range[0]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        # Scanning for files in given directories and loading images. 
        self._set_filesystem(args.dir_data)   
        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0:
            self.images_hr, self.images_lr = list_hr, list_lr
        else: 
            raise ValueError("Invalid file extension %s !" % str(args.ext))

    # =========================================================================
    # Handling the filesystem 
    # =========================================================================
    def _scan(self) -> Tuple[List[str], List[str]]:
        """ Scan given lists of directories for HR and LR images and return 
        list of HR and LR absolute file paths. """
        names_hr = sorted(
            glob.glob(self.dir_hr + "/*" + self.ext[0])
        )
        # For testing issues check if scale == 1, then just return HR images. 
        if self.scale == 1: 
            return names_hr, names_hr
        # Otherwise build LR image names for every HR image. 
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(self.dir_lr + "/{}x{}{}".format(
                filename, self.scale, self.ext[1]
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

    def _load_file(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img':
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        else: 
            raise ValueError("Invalid file extension %s !" % str(self.args.ext))

        def _expand_dimension(img: np.ndarray) -> np.ndarray: 
            if img.ndim == 2 and self.args.n_colors == 3: 
                img = np.stack((img,img,img), axis=2)
            return img

        lr, hr = _expand_dimension(lr), _expand_dimension(hr)
        assert hr.shape[2] == lr.shape[2]

        def _cut_uneven(img: np.ndarray) -> np.ndarray:
            if not img.shape[0] % 2 == 0: 
                img = img[:-1,:,:]
            if not img.shape[1] % 2  == 0: 
                img = img[:,:-1,:]
            return img

        hr =  _cut_uneven(hr)

        if not hr.shape[0] == self.scale*lr.shape[0]: 
            print(f_hr)
            print(hr.shape)
            print(lr.shape)

        assert hr.shape[0] == self.scale*lr.shape[0]
        assert hr.shape[1] == self.scale*lr.shape[1]

        return lr, hr, filename

    # =========================================================================
    # Getter for images
    # =========================================================================
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        # Load image file. 
        lr, hr, filename = self._load_file(idx)
        # Cut patches from file. 
        def _get_patch(lr: np.ndarray, hr: np.ndarray, 
                       scale: int, patch_size: int, do_train: bool):
            if do_train:
                lh, lw = lr.shape[:2]
                hp = patch_size
                lp = hp // scale
                lx = random.randrange(0, lw - lp + 1)
                ly = random.randrange(0, lh - lp + 1)
                hx, hy = scale * lx, scale * ly
                lr = lr[ly:ly + lp, lx:lx + lp, :]
                hr = hr[hy:hy + hp, hx:hx + hp, :]
            else:
                ih, iw = lr.shape[:2]
                hr = hr[0:ih * scale, 0:iw * scale]
            return lr, hr

        patch_size = self.args.patch_size
        assert patch_size <= hr.shape[0] and patch_size <= hr.shape[1]
        pair = _get_patch(lr, hr, self.scale, patch_size, self.train)
        # Normalize patches from rgb_range to [norm_min, norm_max].  
        if not self.args.no_normalize:
            pair = [misc.normalize(x, self.args.rgb_range, 
                    self.args.norm_min, self.args.norm_max) for x in pair]
        # Augment patches (if flag is set). 
        if not self.args.no_augment: 
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5

            def _augment(img: np.ndarray) -> np.ndarray:
                if hflip: img = img[:, ::-1, :]
                if vflip: img = img[::-1, :, :]
                if rot90: img = img.transpose(1, 0, 2)
                return img

            pair = [_augment(x) for x in pair]
        # Set right number of channels. 
        def _set_channel(img, n_channels: int) -> np.ndarray:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            c = img.shape[2]
            if n_channels == 1 and c == 3:
                img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
            elif n_channels == 3 and c == 1:
                img = np.concatenate([img] * n_channels, 2)
            return img

        pair = [_set_channel(x, n_channels=self.args.n_colors) for x in pair]
        # Convert to torch tensor and return. 
        def _np2Tensor(img, rgb_range=255) -> Tensor:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)
            return tensor

        pair_t = [_np2Tensor(x, rgb_range=self.args.rgb_range) for x in pair]
        return pair_t[0], pair_t[1], filename

    # =========================================================================
    # Miscellaneous
    # =========================================================================
    def __len__(self) -> int:
        return len(self.images_hr)
    
# =============================================================================
# DATA LOADING CLASS. 
# =============================================================================
class _DataLoader_(DataLoader): 
    """ Pytorch data loader to load dataset with the following input arguments: 
    - batch_size: number of samples in a batch
    - shuffle: should the dataset be shuffled before loading ?
    - num_workers:  how many subprocesses to use for data loading. 
                    0 means that the data will be loaded in the main process.
    - collate_fn: merges a list of samples to form a mini-batch. """

    def __init__(self, dataset, 
                 batch_size, 
                 shuffle=False, 
                 num_workers=0, 
                 collate_fn=default_collate): 
        super(_DataLoader_, self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            collate_fn=collate_fn,
        )

# =============================================================================
# DATA HANDLING CLASS. 
# =============================================================================
class _Data_(object):
    """ Data loading class which allocates all given training and testing
    dataset stated in the input arguments to a loader (and concatenates 
    them for training). The resulting loader_test and loader_train can be 
    used to load batches from the datasets. """

    def __init__(self, args: argparse.Namespace):
        self.loader_valid = []
        self.loader_test = []
        self.loader_train = {}
        # Check input scales, lists have to be a power of 2. 
        if type(args.scales_valid) == int: 
            args.scales_valid = [args.scales_valid]
        assert misc.all_power2(args.scales_valid)
        if type(args.scales_train) == int: 
            args.scales_train = [args.scales_train]
        assert misc.all_power2(args.scales_train)
        # Load validation dataset. In order to get seperated testing results, 
        # from each dataset (due to comparability reasons) the testing 
        # datasets are each loaded individually. 
        if type(args.data_valid) == str: 
            args.data_valid = [args.data_valid]
        for dataset in args.data_valid:
            for s in args.scales_valid:  
                vset = self.load_dataset(args, dataset, train=False, scale=s)
                self.loader_valid.append(_DataLoader_(vset, 1))            
        if args.valid_only: return
        # Load testing dataset(s). 
        if type(args.data_test) == str: 
            args.data_test = [args.data_test]
        for s in args.scales_train: 
            for dataset in args.data_test:
                tset = self.load_dataset(args, dataset, train=False, scale=s)
                self.loader_test.append(_DataLoader_(tset, 1))
        # Load training dataset, if not testing only. For training several
        # datasets are trained in one process and therefore, each given 
        # training dataset is concatinated to one large dataset (for each scale).
        for s in args.scales_train: 
            tset = self.load_dataset(args, dataset, train=True, scale=s)
            self.loader_train[s] = _DataLoader_(
                tset, args.batch_size, shuffle=True, num_workers=args.n_threads
            )

    @staticmethod 
    def load_dataset(args, name: str, train: bool, scale: int) -> _Dataset_: 
        """ Load dataset from module (in datasets directory). Every module loaded
        should inherit from the _Dataset_ class defined below. """
        m = importlib.import_module("tar.datasets." + name.lower())
        return getattr(m, name)(args, train=train, scale=scale)

